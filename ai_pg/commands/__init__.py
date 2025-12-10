import click

from . import generate_metadata
from . import run_mcp


def get_commands() -> list[click.Command]:
    return [
        generate_metadata.generate_metadata,
        run_mcp.run_mcp,
    ]
