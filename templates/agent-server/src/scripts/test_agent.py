#!/usr/bin/env python
"""
Test an agent via API.

Usage:
    test-agent <agent> <input-json> [--max-iterations N] [--timeout N] [--api URL]

Examples:
    test-agent hello '{"goal": "say hello"}'  # Spawn local server
    test-agent hello '{"goal": "say hello"}' --api=api:3030  # Use running server at api:3030
    test-agent hello '{"goal": "say hello"}' --api=http://example.com:8080  # Use external server
"""

import argparse
import asyncio
import json
import sys
import os
import time
import logging


class ColorFormatter(logging.Formatter):
    """Custom formatter that uses colors for log levels."""

    colors = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[34m",  # Blue
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    prefixes = {
        "DEBUG": "DEBUG: ",
        "INFO": "",  # No prefix for INFO
        "WARNING": "WARNING: ",
        "ERROR": "ERROR: ",
        "CRITICAL": "CRITICAL: ",
    }
    reset = "\033[0m"

    def format(self, record):
        color = self.colors.get(record.levelname, "")
        prefix = self.prefixes.get(record.levelname, "")
        return f"{color}{prefix}{record.getMessage()}{self.reset}"


# Set up logger with color formatter
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_thought(iteration: int, event: dict, verbose: bool = False):
    """Log a thought event with pretty-printed data."""
    if verbose:
        logger.info(f"\nThought (iteration {iteration}):")
        thought_data = {
            "reasoning": event.get("reasoning"),
            "goal_achieved": event.get("goal_achieved"),
            "next_action_needed": event.get("next_action_needed"),
        }
        thought_data = {k: v for k, v in thought_data.items() if v is not None}
        logger.info(json.dumps(thought_data, indent=2))
    else:
        reasoning = event.get("reasoning", "")
        if len(reasoning) > 80:
            reasoning = reasoning[:77] + "..."
        logger.info(f"Iteration {iteration}: {reasoning}")


def log_action(iteration: int, tool: str, event: dict, verbose: bool = False):
    """Log an action event with pretty-printed data."""
    if verbose:
        logger.info(f"\nAction (iteration {iteration}): {tool}")
        action_data = {
            "parameters": event.get("parameters"),
            "result": event.get("result"),
            "success": event.get("success"),
        }
        action_data = {k: v for k, v in action_data.items() if v is not None}
        if action_data:
            logger.info(json.dumps(action_data, indent=2))
    else:
        success = "✓" if event.get("success", True) else "✗"
        logger.info(f"Iteration {iteration}: {tool} {success}")


def log_success(result: dict, execution_time: float):
    """Log success with pretty-printed result."""
    logger.info(f"\n\033[1mSuccess! ({execution_time:.2f}s)\033[0m")
    if result:
        logger.info(json.dumps(result, indent=2))


async def monitor_call_via_websocket(
    base_url: str, call_id: str, timeout: float, verbose: bool = False
):
    """Monitor call progress via WebSocket."""
    import websockets

    # Convert http URL to ws URL
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_endpoint = f"{ws_url}/api/v1/calls/{call_id}/events/stream"

    logger.debug(f"Opening WebSocket to monitor call {call_id}")

    events_received = []
    final_status = None
    final_result = None
    final_error = None

    try:
        async with websockets.connect(ws_endpoint) as websocket:
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    data = json.loads(message)

                    if data.get("type") == "event":
                        event = data.get("data", {})
                        event_type = event.get("event_type", "unknown")
                        events_received.append(event_type)

                        if event_type == "thought":
                            iteration = event.get("iteration", "?")
                            log_thought(iteration, event, verbose)
                        elif event_type == "action":
                            iteration = event.get("iteration", "?")
                            tool = event.get("tool_name", "unknown")
                            log_action(iteration, tool, event, verbose)
                        elif event_type == "result":
                            final_result = event.get("result", {})
                        elif event_type == "status_change":
                            new_status = event.get("new_status", "unknown")
                            final_status = new_status
                            logger.debug(f"Status changed to: {new_status}")
                            if new_status in ["completed", "failed", "cancelled"]:
                                break
                        elif event_type == "error":
                            final_error = event.get("error_message", "Unknown error")
                            logger.error(f"Agent error: {final_error}")
                            break
                        else:
                            logger.debug(f"Event: {event_type}")
                    elif data.get("type") == "error":
                        error_msg = data.get("data", {}).get(
                            "error_message", "Unknown error"
                        )
                        logger.error(f"WebSocket error: {error_msg}")
                        final_error = error_msg
                        break

                except asyncio.TimeoutError:
                    logger.error(f"WebSocket timeout after {timeout}s")
                    final_status = "timeout"
                    break
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    break

    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        final_error = str(e)

    return final_status, final_result, final_error


def main():
    parser = argparse.ArgumentParser(description="Test an agent")
    parser.add_argument("agent", help="Agent name")
    parser.add_argument("input", help="Input JSON")
    parser.add_argument(
        "--max-iterations", type=int, default=3, help="Max iterations (default: 3)"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Timeout seconds (default: 60)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed event data"
    )
    parser.add_argument(
        "--api",
        default=None,
        help="Use external API at URL (default: spawn local server)",
    )

    args = parser.parse_args()

    # Set log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Parse input
    try:
        input_data = json.loads(args.input)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        logger.error("TEST FAILED: Invalid input JSON")
        sys.exit(1)

    # Determine mode
    if args.api:
        base_url = args.api if args.api.startswith("http") else f"http://{args.api}"
        mode = "external"
    else:
        base_url = "http://localhost:3030"
        mode = "local"

    call_msg = f"Calling {args.agent}({json.dumps(input_data, separators=(',', ':'))})"
    params = []
    if not args.api:
        params.append("local")
    else:
        params.append(f"api={base_url}")
    if args.max_iterations != 3:
        params.append(f"max_iterations={args.max_iterations}")
    if args.timeout != 60.0:
        params.append(f"timeout={args.timeout}s")

    if params:
        logger.info(f"{call_msg} \033[2m[{', '.join(params)}]\033[0m")
    else:
        logger.info(call_msg)
    import httpx

    start_time = time.time()

    try:
        server_proc = None
        if not args.api:
            import subprocess
            import atexit

            env = os.environ.copy()
            env["USE_POSTGRES"] = "false"
            env["LOG_LEVEL"] = "WARNING"
            env["HTTP_DEBUG"] = "true"

            server_proc = subprocess.Popen(
                ["./bin/uv-run", "app"], env=env, stdout=sys.stderr, stderr=sys.stderr
            )

            def cleanup():
                if server_proc and server_proc.poll() is None:
                    server_proc.terminate()
                    try:
                        server_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        server_proc.kill()

            atexit.register(cleanup)

            for i in range(50):  # 5 seconds max
                try:
                    response = httpx.get(f"{base_url}/api/v1/agents")
                    if response.status_code == 200:
                        logger.debug("Server ready")
                        break
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                logger.error(f"Server failed to start at {base_url} after 5 seconds")
                logger.error("TEST FAILED: Server startup timeout")
                sys.exit(1)

        with httpx.Client() as client:
            call_spec = {"input_data": input_data}
            if args.max_iterations:
                call_spec["max_iterations"] = args.max_iterations
            endpoint = f"{base_url}/api/v1/agents/{args.agent}/calls"
            try:
                response = client.post(endpoint, json=call_spec, timeout=5.0)
                try:
                    response_body = response.json()
                except:
                    response_body = response.text

                if response.status_code >= 400:
                    logger.error(f"HTTP {response.status_code} from POST {endpoint}")
                    logger.error(f"Request body: {json.dumps(call_spec)}")
                    logger.error(
                        f"Response body: {json.dumps(response_body) if isinstance(response_body, dict) else response_body}"
                    )
                    if isinstance(response_body, dict):
                        detail = response_body.get("detail", "No error detail provided")
                    else:
                        detail = (
                            response_body[:500] if response_body else "No response body"
                        )

                    logger.error(
                        f"TEST FAILED: POST {endpoint} returned HTTP {response.status_code} - {detail}"
                    )
                    sys.exit(1)

                result = response_body

            except httpx.TimeoutException:
                logger.error(f"Request timeout calling POST {endpoint}")
                logger.error("TEST FAILED: HTTP request timeout")
                sys.exit(1)
            except httpx.ConnectError as e:
                logger.error(f"Connection error calling POST {endpoint}: {e}")
                logger.error("TEST FAILED: Could not connect to server")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Unexpected error calling POST {endpoint}: {e}")
                logger.error(f"TEST FAILED: {type(e).__name__}: {e}")
                sys.exit(1)

            call_id = result.get("id")
            if not call_id:
                logger.error(f"No call ID in response: {result}")
                logger.error("TEST FAILED: Invalid server response (no call ID)")
                sys.exit(1)

            logger.debug(f"Call ID: {call_id}")

        # Monitor via WebSocket
        final_status, final_result, final_error = asyncio.run(
            monitor_call_via_websocket(base_url, call_id, args.timeout, args.verbose)
        )

        if final_status == "completed":
            execution_time = time.time() - start_time
            log_success(final_result, execution_time)
            sys.exit(0)
        elif final_status == "failed" or final_error:
            error_msg = final_error or "Unknown error"
            logger.error(f"Agent failed: {error_msg}")
            sys.exit(1)
        elif final_status == "cancelled":
            logger.error("Agent cancelled")
            sys.exit(1)
        elif final_status == "timeout":
            logger.error(f"Timeout after {args.timeout}s")
            sys.exit(2)
        else:
            logger.error(f"Unexpected status: {final_status}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("")
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
