# Qdrant Hybrid Search and Summarization Toolkit

## Description

This project provides Python tools to interact with a Qdrant vector database for advanced search and summarization tasks. It leverages OpenAI-compatible APIs for generating embeddings, reranking search results, and performing text summarization. It also utilizes FastEmbed for creating sparse vectors.

The toolkit consists of two main components:

1.  **`retriever.py`**: Performs hybrid search (combining dense and sparse vectors with Reciprocal Rank Fusion - RRF) on specified Qdrant collections. It retrieves relevant documents based on a query, reranks them using a dedicated reranker model for improved relevance, and can optionally summarize the top results.
2.  **`summarizer.py`**: Retrieves *all* documents from a specified Qdrant collection and generates a comprehensive summary. It automatically chooses between a direct summarization approach (if the total text fits within the model's context window) or a map-reduce strategy (summarizing chunks individually and then summarizing the summaries) for very large collections.

## Features

*   **Hybrid Search:** Combines dense and sparse vector search using Qdrant's RRF fusion.
*   **Reranking:** Improves search result relevance using an external reranker model via an OpenAI-compatible API.
*   **Targeted Summarization:** Summarizes the top results from a hybrid search, focusing on the query context (`retriever.py`).
*   **Full Collection Summarization:** Summarizes the entire content of a Qdrant collection (`summarizer.py`).
*   **Adaptive Summarization Strategy:** Automatically uses direct or map-reduce summarization based on estimated token count (`summarizer.py`).
*   **Configurable:** Uses an `__init__.py` file (or similar configuration mechanism) to manage API endpoints, keys, model names, and Qdrant connection details.
*   **Error Handling:** Includes basic error handling and logging for API calls and Qdrant interactions.

## Getting Started

### Prerequisites

*   Python 3.10+
*   Access to a running Qdrant instance.
*   Access to an OpenAI-compatible API endpoint (e.g., OpenAI API, a local LLM server like Ollama or LM Studio) for embeddings, reranking, and summarization.
*   Required Python packages (see Dependencies).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FBR65/summarize
    cd summarize
    ```
2.  **Create `__init__.py`:**
    Create a file named `__init__.py` in the project's root directory and define the necessary configuration constants. Example:
    ```python
    # __init__.py

    # Qdrant Configuration
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_TIMEOUT = 120

    # OpenAI-Compatible API Configuration
    BASE_URL = "http://localhost:11434/v1" # Example for local Ollama
    API_KEY = "ollama" # Or your actual API key if required

    # Model Names (Adjust based on your API provider/local models)
    EMBEDDING_MODEL = "paraphrase-multilingual:latest" # Dense embedding model
    SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions" # Sparse embedding model (ensure compatibility with FastEmbed)
    RERANKER_MODEL = "bge-m3:latest" # Reranker model (often requires specific API format)
    SUMMARIZATION_MODEL = "granite3.3:8b" # Chat/Summarization model

    # Summarizer Configuration
    MODEL_CONTEXT_LIMIT_TOKENS = 8192 # Context window size of your summarization model
    ```
3.  **Install Dependencies:**
    Then install them using `uv`:
    ```bash
    # Ensure you have uv installed (https://github.com/astral-sh/uv)
    # It's often recommended to use a virtual environment
    uv sync
    ```

## Usage

### 1. Hybrid Search and Targeted Summarization (`retriever.py`)

This script performs a hybrid search and can summarize the top results.

*   **Modify `if __name__ == "__main__":` block:**
    *   Set `collection_name_prefix` to the prefix used for your Qdrant collections (e.g., `"my_docs"` if your collections are `"my_docs_COSINE"`).
    *   Set `search_query` to your desired query.
    *   Adjust `limit` to control the number of results retrieved and potentially summarized.
*   **Run the script:**
    ```bash
    python retriever.py
    ```
    The script will output the raw and reranked results per collection, the final deduplicated and reranked results, and a summary of the top documents based on the query.

### 2. Full Collection Summarization (`summarizer.py`)

This script retrieves all documents from a *single, specific* collection and summarizes them.

*   **Modify `if __name__ == "__main__":` block:**
    *   Set `target_collection_name` to the *exact* name of the Qdrant collection you want to summarize (e.g., `"my_docs_COSINE"`).
*   **Run the script:**
    ```bash
    python summarizer.py
    ```
    The script will fetch all documents, estimate the token count, choose a summarization strategy (direct or map-reduce), and print the final summary. Be aware that fetching and summarizing large collections can take significant time and resources.

## Configuration (`__init__.py`)

The following constants need to be defined in `__init__.py`:

*   `QDRANT_HOST`: Hostname of your Qdrant instance.
*   `QDRANT_PORT`: Port of your Qdrant instance.
*   `QDRANT_TIMEOUT`: (Optional) Client timeout in seconds.
*   `BASE_URL`: Base URL for the OpenAI-compatible API.
*   `API_KEY`: API key for the service (might be ignored by some local servers).
*   `EMBEDDING_MODEL`: Name/identifier of the dense embedding model.
*   `SPARSE_MODEL`: Name/identifier of the sparse embedding model (compatible with FastEmbed).
*   `RERANKER_MODEL`: Name/identifier of the reranker model.
*   `SUMMARIZATION_MODEL`: Name/identifier of the chat/summarization model.
*   `MODEL_CONTEXT_LIMIT_TOKENS`: The maximum context window size (in tokens) for the `SUMMARIZATION_MODEL`.

## Dependencies

*   openai: For interacting with OpenAI-compatible APIs.
*   qdrant-client: For interacting with the Qdrant vector database.
*   fastembed: For generating sparse vector embeddings.
*   pydantic: For data validation (used in `retriever.py`).
*   tiktoken: (Optional) For accurate token counting, improving the summarization strategy choice.

## License

AGPLv3 as written in LICENSE.md 
