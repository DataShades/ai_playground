import os


class Config:
    # Ollama Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
    OLLAMA_THINKING = os.getenv("OLLAMA_THINKING", "False").lower() in (
        "true",
        "1",
        "t",
    )
    # Temperature controls randomness of outputs: 0.0 is deterministic, 1.0 is creative
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    # Default context_window is -1 (infer from model)
    # Increasing the context window allows the model to consider more prior conversation
    # For our case, if the data sample + schema are large, we may need a bigger context window
    # otherwise the model starts to hallucinate. For example, it took the data sample in
    # consideration, but partially ignored the schema when generating metadata.
    OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "38192"))

    # MCP Settings
    MCP_CLIENT_URL = os.getenv("MCP_CLIENT_URL", "http://127.0.0.1:5337/sse")

    # Agent Settings
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))
