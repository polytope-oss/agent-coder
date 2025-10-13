"""
Agent registry for managing and tracking agent instances.
"""

import asyncio
from typing import Any
from uuid import UUID
from pydantic import ValidationError
from .utils import log
from .models.calls import CallThought, CallAction, CallResult, CallError, CallStatus
from .repositories.calls import CallRepository
from . import conf

logger = log.get_logger(__name__)


class AgentRegistry:
    """Registry for tracking and managing agents."""

    def __init__(self, call_repository: CallRepository):
        self.agents: dict[str, Any] = {}
        self.call_repository = call_repository
        self._call_agents: dict[UUID, str] = {}  # call_id -> agent_name mapping
        self._running_tasks: dict[
            UUID, asyncio.Task
        ] = {}  # call_id -> asyncio.Task mapping for cancellation
        self._event_queue: asyncio.Queue = (
            asyncio.Queue()
        )  # Queue for events from Opper SDK
        self._worker_task: asyncio.Task | None = None  # Event worker task

    def register_agent(self, name: str, agent: Any):
        """Register an agent in the registry."""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    async def _handle_thought_created(
        self, call_id: UUID, data: dict[str, Any]
    ) -> None:
        """
        Handle thought_created event from Opper SDK.

        Data structure: {"iteration": N, "thought": Thought.dict()}
        """
        # Process thought silently
        thought_dict = data.get("thought", {})
        iteration = data.get("iteration", 0)

        thought = CallThought(
            call_id=call_id,
            iteration=iteration,
            reasoning=thought_dict.get("reasoning", ""),
            goal_achieved=thought_dict.get("goal_achieved", False),
            todo_list=thought_dict.get("todo_list"),
            next_action_needed=thought_dict.get("next_action_needed", False),
            # Extended fields
            tool_name=thought_dict.get("tool_name"),
            tool_parameters=thought_dict.get("tool_parameters"),
            expected_outcome=thought_dict.get("expected_outcome"),
            user_message=thought_dict.get("user_message"),
            # Store raw data
            raw_data=data,
        )
        await self.call_repository.append_call_thought(call_id, thought)

    async def _handle_action_executed(
        self, call_id: UUID, data: dict[str, Any]
    ) -> None:
        """
        Handle action_executed event from Opper SDK.

        Data structure: {"iteration": N, "thought": {...}, "action_result": {...}}
        """
        # Process action silently
        action_dict = data.get("action_result", {})
        iteration = data.get("iteration", 0)

        result = action_dict.get("result")
        success = action_dict.get("success", True)
        error_message = action_dict.get("error_message")

        # Parse stringified Python dicts if needed
        if isinstance(result, str) and result.strip().startswith("{"):
            try:
                import ast

                parsed_result = ast.literal_eval(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                    # Extract success/error info from the parsed dict
                    if "success" in result:
                        success = result["success"]
                    if "error" in result and not error_message:
                        error_message = result["error"]
            except (ValueError, SyntaxError):
                # Keep original string if parsing fails
                pass

        action = CallAction(
            call_id=call_id,
            iteration=iteration,
            tool_name=action_dict.get("tool_name", "unknown"),
            parameters=action_dict.get("parameters", {}),
            result=result,
            success=success,
            execution_time=action_dict.get("execution_time", 0.0),
            error_message=error_message,
            # Store raw data
            raw_data=data,
        )
        await self.call_repository.append_call_action(call_id, action)

    async def _handle_goal_completed(self, call_id: UUID, data: dict[str, Any]) -> None:
        """
        Handle goal_completed event from Opper SDK.

        Data structure: {"goal": str, "achieved": bool, "iterations": N, "final_result": Any}
        """
        iterations = data.get("iterations")
        final_result = data.get("final_result")

        # Build metadata
        metadata = {}
        if iterations is not None:
            metadata["iterations"] = iterations
        if data.get("goal"):
            metadata["goal"] = data["goal"]

        # Try to extract structured fields if final_result is a dict
        executive_summary = None
        key_findings = None
        citations = None
        if isinstance(final_result, dict):
            executive_summary = final_result.get("executive_summary")
            key_findings = final_result.get("key_findings")
            citations = final_result.get("citations")

        try:
            result = CallResult(
                call_id=call_id,
                success=data.get("achieved", False),  # Default should be False, not True
                result=final_result,
                executive_summary=executive_summary,
                key_findings=key_findings,
                citations=citations,
                metadata=metadata or None,
                raw_data=data,
            )
            await self.call_repository.register_call_done(call_id, result)
        except ValidationError as e:
            # Agent output doesn't match expected schema
            debug_mode = conf.get_debug_mode()

            if debug_mode:
                # In debug mode, include full validation details
                error_message = "Agent output validation failed: " + str(e)
                error_details = {
                    "validation_errors": [
                        {
                            "field": ".".join(str(loc) for loc in err["loc"]),
                            "type": err["type"],
                            "message": err["msg"],
                        }
                        for err in e.errors()
                    ],
                    "raw_output": data.get("final_result"),
                }
            else:
                # In production mode, use generic message
                error_message = "Agent produced output that doesn't match the expected schema"
                error_details = None

            # Mark the call as failed
            error = CallError(
                call_id=call_id,
                error_type="output_validation_error",
                error_message=error_message,
                error_details=error_details,
                recoverable=False,
                raw_data=data,
            )
            await self.call_repository.register_call_error(call_id, error)
            logger.error(f"Output validation error for call {call_id}: {error_message}")

    async def _handle_error_event(self, call_id: UUID, data: Any) -> None:
        """
        Handle error events.
        """
        if isinstance(data, dict):
            error_type = data.get("error_type", "execution_error")
            error_message = str(data.get("error_message", data))
            recoverable = data.get("recoverable", False)
            raw_data = data
        else:
            error_type = "execution_error"
            error_message = str(data)
            recoverable = False
            raw_data = {"error": str(data)}
        
        error = CallError(
            call_id=call_id,
            error_type=error_type,
            error_message=error_message,
            recoverable=recoverable,
            raw_data=raw_data,
        )
        await self.call_repository.register_call_error(call_id, error)

    def _create_event_handler(self, agent_name: str, call_id: UUID):
        """
        Create an event handler for a specific agent execution.

        Returns a callback function that captures the call_id.
        """

        def event_handler(event_type: str, data: Any):
            """
            Handle events from the Opper agent execution.

            Event types from Opper SDK (base_agent.py):
            - thought_created: Contains {"iteration": N, "thought": {...}}
            - action_executed: Contains {"iteration": N, "thought": {...}, "action_result": {...}}
            - goal_completed: Contains {"goal": str, "achieved": bool, "iterations": N, "final_result": Any}
            - error: Error occurred during execution

            TODO: Make this async when Opper SDK supports async callbacks
            """
            # Log only important events
            logger.debug(
                f"Received event '{event_type}' for call {call_id} from agent {agent_name}"
            )

            try:
                # Just put the event on the queue (non-blocking)
                self._event_queue.put_nowait((event_type, call_id, data))
            except asyncio.QueueFull:
                logger.error(
                    f"Event queue is full! Dropping event {event_type} for call {call_id}"
                )
            except Exception as e:
                logger.error(f"Error queueing event {event_type}: {e}", exc_info=True)

        return event_handler

    def get_agent(self, name: str) -> Any | None:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> dict[str, dict[str, Any]]:
        """List all registered agents with their metadata and schemas."""
        from .utils.agent_utils import get_agent_schema

        result = {}
        for name, agent in self.agents.items():
            schema_info = get_agent_schema(agent)
            result[name] = {
                "name": agent.name if hasattr(agent, "name") else name,
                "description": agent.description
                if hasattr(agent, "description")
                else "",
                "max_iterations": agent.max_iterations
                if hasattr(agent, "max_iterations")
                else 10,
                "verbose": agent.verbose if hasattr(agent, "verbose") else False,
                "tools": [tool.name for tool in agent.tools]
                if hasattr(agent, "tools")
                else [],
                "input_schema": schema_info["input_schema"],
                "output_schema": schema_info["output_schema"],
            }
        return result

    async def execute_agent(
        self,
        agent_name: str,
        call_id: UUID,
        input_data: dict[str, Any],
        max_iterations: int | None = None,
    ) -> Any:
        """Execute an agent with the given input data."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        # Set max_iterations if provided
        original_max = None
        if max_iterations is not None:
            original_max = agent.max_iterations
            agent.max_iterations = max_iterations
            logger.debug(f"Set max_iterations to {max_iterations} for call {call_id}")

        # Track this call-agent association
        self._call_agents[call_id] = agent_name

        # Set up event handler for this specific call
        original_callback = getattr(agent, "callback", None)
        event_handler = self._create_event_handler(agent_name, call_id)
        agent.callback = event_handler
        logger.debug(f"Set up event handler for agent {agent_name}, call {call_id}")

        try:
            # Register call as started
            await self.call_repository.register_call_started(call_id)

            # Run agent.process() in a thread pool to avoid blocking the main event loop
            def run_agent_in_thread():
                """Run the async agent.process() in a new event loop in a thread."""

                # Use asyncio.run() which creates a new event loop, runs the coroutine,
                # and cleans up properly. This is thread-safe for parallel calls.
                async def run_agent_async():
                    return await agent.process(input_data)

                return asyncio.run(run_agent_async())

            # Run in thread pool to avoid blocking the main FastAPI event loop
            result = await asyncio.get_running_loop().run_in_executor(
                None, run_agent_in_thread
            )

            # Check if call was cancelled
            call = await self.call_repository.get_call(call_id)
            if call and call.status == CallStatus.CANCELLED:
                logger.info(f"Call {call_id} was cancelled during execution")
                return None

            return result

        except asyncio.CancelledError:
            # Handle cancellation
            logger.info(f"Call {call_id} execution cancelled")
            await self.call_repository.register_call_cancelled(call_id)
            raise
        except Exception as e:
            # Register error
            error = CallError(
                call_id=call_id,
                error_type="execution_error",
                error_message=str(e),
                recoverable=False,
            )
            await self.call_repository.register_call_error(call_id, error)
            raise
        finally:
            # Restore original callback
            if original_callback:
                agent.callback = original_callback
            elif hasattr(agent, "callback"):
                delattr(agent, "callback")

            # Restore original max_iterations if we changed it
            if original_max is not None:
                agent.max_iterations = original_max

            # Clean up execution tracking
            self._call_agents.pop(call_id, None)
            # Remove from running tasks
            self._running_tasks.pop(call_id, None)

    def start_agent_execution(
        self,
        agent_name: str,
        call_id: UUID,
        input_data: dict[str, Any],
        max_iterations: int | None = None,
    ) -> asyncio.Task:
        """
        Start agent execution in a background task.

        Returns the task handle for potential cancellation.
        """
        task = asyncio.create_task(
            self.execute_agent(agent_name, call_id, input_data, max_iterations)
        )
        self._running_tasks[call_id] = task
        return task

    async def cancel_agent_execution(self, call_id: UUID) -> bool:
        """
        Cancel a running agent execution.

        Returns True if cancellation was initiated, False if task not found.
        """
        task = self._running_tasks.get(call_id)
        if task and not task.done():
            task.cancel()
            # Wait briefly for task to acknowledge cancellation
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            return True
        return False

    async def _event_worker(self):
        """Long-running worker that processes all events from the queue."""
        logger.info("Event worker started")
        while True:
            try:
                # Wait for an event from any agent
                event_type, call_id, data = await self._event_queue.get()
                logger.debug(f"Event worker processing {event_type} for call {call_id}")

                # Process the event based on type
                if event_type == "thought_created":
                    await self._handle_thought_created(call_id, data)
                elif event_type == "action_executed":
                    await self._handle_action_executed(call_id, data)
                elif event_type == "goal_completed":
                    await self._handle_goal_completed(call_id, data)
                elif event_type == "goal_start":
                    # Just log it, no action needed
                    logger.debug(f"Goal started for call {call_id}")
                elif event_type == "error":
                    await self._handle_error_event(call_id, data)
                else:
                    logger.warning(f"Event worker: unhandled event type '{event_type}'")

                # Mark task as done
                self._event_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Event worker shutting down")
                break
            except Exception as e:
                logger.error(f"Error in event worker: {e}", exc_info=True)
                # Continue processing other events

    async def start_event_worker(self):
        """Start the event worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._event_worker())
            logger.info("Started event worker task")

    async def stop_event_worker(self):
        """Stop the event worker task."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped event worker task")

    async def cleanup(self):
        """Clean up any remaining tasks."""
        # Stop the event worker
        await self.stop_event_worker()

        # Cancel all running agent tasks
        for task in self._running_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for them to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        self._running_tasks.clear()
