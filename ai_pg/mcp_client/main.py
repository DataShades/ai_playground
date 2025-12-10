from pathlib import Path
from typing import Any

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP(name="CKAN MCP")


@mcp.tool
async def get_metadata_schema() -> dict[str, Any]:
    """Returns the required schema/format for dataset metadata.

    ALWAYS call this FIRST before generating metadata to ensure correct format.

    Returns:
        dict[str, Any]: A dictionary representing the metadata schema.
    """
    return {
        "title": "string",
        "notes": "string",
        "publisher": {"name": "string", "email": "string"},
        "tags": [{"name": "string"}],
        "license": "string",
        "organization": {"name": "string", "title": "string"},
    }


@mcp.tool
async def get_resource_data(name: str) -> dict[str, Any]:
    """Fetches data from a CSV file including columns and sample rows.

    Call this AFTER getting the schema to retrieve the actual file data.

    Args:
        name: The filename to fetch data from
    Args:
        name (str): The name of the resource file (CSV).

    Returns:
        dict[str, Any]: A dictionary containing column names and a sample of the data.

    Example:
        >>> await get_resource_data("data.csv")
        {
            "columns": ["id", "name", "value"],
            "sample": "   id   name  value\n0   1  Alice    10\n1   2    Bob    20\n2   3  Carol    30"
        }
    """
    if not name.endswith(".csv"):
        name += ".csv"

    file_path = Path(__file__) / "data" / name
    df = pd.read_csv(file_path)

    return {
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_string(),
    }


if __name__ == "__main__":
    # mcp.run(transport="streamable-http", host="127.0.0.1", port=5337, path="/mcp")
    mcp.run(transport="sse", host="127.0.0.1", port=5337)
