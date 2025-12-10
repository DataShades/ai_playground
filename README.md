# AI Playground (ai_pg)

This project is a playground for testing the integration of LLM capabilities into CKAN. It utilizes **Llama Index**, **Ollama**, and **FastMCP** to demonstrate how to generate metadata from data files.

## Prerequisites

Before setting up the project, you must ensure that **Ollama** is installed and the required model is available.

### 1. Install Ollama
Download and install [Ollama](https://ollama.com/) for your operating system.

### 2. Pull the Required Model
This project uses the `qwen3:8b` model. Once Ollama is installed, open your terminal and run:

```bash
ollama pull qwen3:8b
```

## Installation

1. Clone this repository and navigate to the project directory.
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install the project:
    ```bash
    pip install -e .
    ```

## Usage

This tool provides two main CLI commands to interact with the LLM and the MCP server.

### 1. Run the MCP Server

The MCP (Model Context Protocol) server needs to be running, as it provides `tools` for the agent.

```bash
ai_pg run-mcp
```

This command starts the local MCP server located in `mcp_client/main.py`.

### 2. Generate Metadata

This command queries the local Ollama LLM and the MCP server to generate metadata for a specific dataset file.

**Note:** The file you specify must exist in the `data` folder. You can also specify a file from your file system, but stick to the CSV format, as other formats are not yet supported.

```bash
ai_pg generate-metadata -f ./data/titanic.csv
```
