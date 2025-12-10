import click
import subprocess
import sys
from pathlib import Path


@click.command(name="run-mcp")
def run_mcp():
    """Run the local MCP server from mcp_client/main.py"""
    project_root = Path(__file__).resolve().parents[1]
    main_path = project_root / "mcp_client" / "main.py"

    if not main_path.exists():
        click.echo(f"Error: MCP main.py not found at {main_path}")
        sys.exit(1)

    click.echo("Starting local MCP server...")

    # Run it as a subprocess so it can block and print logs
    subprocess.run([sys.executable, str(main_path)])
