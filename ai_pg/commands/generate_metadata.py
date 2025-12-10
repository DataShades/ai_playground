import asyncio
import time
from functools import wraps

import click
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.workflow import Context, WorkflowRuntimeError
from llama_index.llms.ollama import Ollama
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from ollama import ResponseError

from ai_pg.config import Config


SYSTEM_PROMPT = [
    "Generate dataset metadata following these steps:",
    "1. Call get_metadata_schema (get format)",
    "2. Call get_resource_data with filename (get data)",
    "3. Return JSON matching schema",
    "Output only valid JSON, no other text.",
]


def sync(func):
    """Decorator that wraps coroutine with asyncio.run."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@click.command(name="generate-metadata")
@click.option("-f", "--filename", type=str, required=True)
@sync
async def generate_metadata(filename: str) -> None:
    """Generate dataset metadata using an agent with MCP tools."""
    mcp_client = BasicMCPClient(Config.MCP_CLIENT_URL)
    mcp_tools = McpToolSpec(client=mcp_client)

    agent = await get_agent(mcp_tools, SYSTEM_PROMPT)
    agent_context = Context(agent)

    user_input = (
        f'Generate dataset metadata for "{filename}". '
        "First check the metadata schema, then fetch the file data, "
        "and create metadata matching the schema."
    )

    response = await handle_user_message(user_input, agent, agent_context)

    click.secho(response, fg="green")


async def get_agent(
    tools: McpToolSpec, system_prompt: list[str]
) -> FunctionAgent | ReActAgent:
    available_tools = await tools.to_tool_list_async()

    click.secho(
        "Creating agent with available tools: "
        + ", ".join(
            tool.metadata.name or str(tool.metadata) for tool in available_tools
        )
        + "\n",
        fg="blue",
    )
    return FunctionAgent(
        tools=available_tools,
        llm=Ollama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            context_window=Config.OLLAMA_CONTEXT_WINDOW,
            request_timeout=Config.OLLAMA_TIMEOUT,
            thinking=Config.OLLAMA_THINKING,
            temperature=Config.OLLAMA_TEMPERATURE,
        ),
        system_prompt="\n".join(system_prompt),
    )


async def handle_user_message(
    message_content: str,
    agent: FunctionAgent | ReActAgent,
    agent_context: Context,
):
    click.secho(f"Handling user message: {message_content} \n", fg="yellow")
    start = time.time()
    tool_times = {}

    handler = agent.run(
        message_content,
        ctx=agent_context,
        max_iterations=Config.AGENT_MAX_ITERATIONS,
        # batch_size=10,  # Allow multiple tool calls in parallel if possible (allegedly, not confirmed)
    )
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            tool_times[event.tool_name] = {"start": time.time()}
            click.secho(
                f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}"
            )
        elif isinstance(event, ToolCallResult):
            if event.tool_name in tool_times:
                elapsed = time.time() - tool_times[event.tool_name]["start"]
                click.secho(
                    f"Tool {event.tool_name} completed in {elapsed:.2f} seconds \n"
                )
            # click.secho(f"Tool {event.tool_name} returned {event.tool_output} \n")

    try:
        response = await handler
    except (WorkflowRuntimeError, ResponseError) as e:
        return "Agent failed to produce a response. Error: " + str(e)

    total_time = time.time() - start
    click.secho(
        f"Completed handling user message in {total_time:.2f} seconds \n", fg="yellow"
    )

    return str(response)
