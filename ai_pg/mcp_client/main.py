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
        dict[str, Any]: A JSON schema representing the metadata schema.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "notes": {"type": "string"},
            "publisher": {
                "type": ["object", "null"],
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name", "email"],
            },
            "tags": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "license": {"type": ["string", "null"]},
            "organization": {
                "type": ["object", "null"],
                "properties": {"name": {"type": "string"}, "title": {"type": "string"}},
                "required": ["name", "title"],
            },
            "update_frequency": {
                "type": ["string", "null"],
                "enum": [
                    "daily",
                    "weekly",
                    "monthly",
                    "quarterly",
                    "biennially",
                    "biannually",
                    "annually",
                    "infrequently",
                    "never",
                ],
            },
            "language": {"type": ["string", "null"]},
            "jurisdiction": {"type": ["string", "null"]},
        },
        "required": ["title", "notes", "tags"],
        "additionalProperties": False,
    }


@mcp.tool
async def get_resource_data(filepath: str) -> dict[str, Any]:
    """Fetches data from a CSV file including columns and sample rows.

    Call this AFTER getting the schema to retrieve the actual file data.

    Args:
        filepath: The filename to fetch data from

    Returns:
        dict[str, Any]: A dictionary containing column names and a sample of the data.

    Example:
        >>> await get_resource_data("data.csv")
        {
            "columns": ["id", "name", "value"],
            "sample": "   id   name  value\n0   1  Alice    10\n1   2    Bob    20\n2   3  Carol    30"
        }
    """
    df = pd.read_csv(Path(filepath))

    return {
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_string(),
    }


if __name__ == "__main__":
    # mcp.run(transport="streamable-http", host="127.0.0.1", port=5337, path="/mcp")
    mcp.run(transport="sse", host="127.0.0.1", port=5337)
