# Agent Coder

A coding agent flavor using the [Opper Agent SDK](https://github.com/opper-ai/opperai-agent-sdk) using [Polytope](https://polytope.com) - lets you vibe-code production-ready AI agents.

## What is this?

It's a set of tools you can use with the Polytope MCP server to make your coding agent great at building AI agents.

The core service is `agent-server`. This is a bare-bones server for hosting AI agents built with the Opper Agent SDK. It includes a set of tools that make it easy for your coding agent and for you to build, run and evaluate agents.

There's also a browser dashboard that lets you inspect and interact with your agents.

## How it works

1. You prompt your coding agent to build some kind of AI agent
2. Your coding agent calls the `add-agent-server` tool to scaffold the agent system
3. The tool instantiates the `agent-server` and `agent-dashboard` in your repo, starts them, and provides a set of tools and context for your coding agent to use
4. Your coding agent now works in a focused way to build and evaluate agents

## Prerequisites

Install Polytope:
```bash
curl -s https://polytope.com/install.sh | sh
```

Get an Opper API key from [https://platform.opper.ai](https://platform.opper.ai).

## Setup Instructions

### Step 1: Include this repository

Include this repository's tools in your project:

```bash
pt include gh:polytope-oss/agent-coder
```

This creates a `polytope.yml` file (unless you already had one) and adds this repository as an include entry.

### Step 2: Set your Opper API key

Set your Opper API key as a secret:

```bash
pt secret set opper-api-key [your-opper-api-key]
```

### Step 3: Start the Polytope MCP server

Run the following command in your project directory:

```bash
pt run --mcp
```

This launches the Polytope terminal UI with a clean sandbox and an MCP server reachable via `http://localhost:31338/mcp`.

### Step 4: Add the Polytope MCP server to your coding agent

The way to do this depends on which coding agent you are using.

#### For Claude Code:
Run
```bash
claude mcp add --transport http polytope http://localhost:31338/mcp
```
in your project directory.

### Step 5: Create your agent

Ask your coding agent to create your agent, for example:

```
Please build a research assistant agent that can investigate any topic by gathering information from multiple sources (Wikipedia, web search), analyzing different perspectives, and synthesizing a comprehensive report with citations.
```
