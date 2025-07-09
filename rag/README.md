# TechniqueRAG Ranking Pipeline

The pipeline employs a two-stage process.

1.  **Initial Retrieval (BM25):** A fast lexical search model (BM25) retrieves a set of potentially relevant techniques from a corpus.
2.  **Re-ranking (Task Adapted RankGPT):** A large language model, prompted as a "RankGPT" security analyst, re-ranks the retrieved techniques for higher precision and relevance.

## Data Formats

### Input Query File

The `--input_file` argument requires a Tab-Separated Values (TSV) file. The file must contain at least two columns: `query` and `tech_id`.

-   `query`: A string containing the textual description of the attack or threat to be analyzed.
-   `tech_id`: A string representation of a Python list of the ground-truth MITRE ATT&CK technique IDs relevant to the query.

**Example `queries.tsv`:**

```tsv
query	tech_id
"The malware uses PowerShell to download and execute a payload from a remote server."	"['T1059.001', 'T1105']"
"An adversary gains initial access by sending a spearphishing email with a malicious attachment."	"['T1566.001']"
```

### Corpus Summaries

The pipeline uses pre-generated summaries for each MITRE technique. These summaries must be located in the `rag/corpus_summaries/` directory.

-   Each summary must be a JSON file named after the technique ID (e.g., `T1059.001.json`).
-   The JSON file format must match the response structure of the OpenAI Chat Completions API.

**Example `T1059.001.json`:**

```json
{
  "choices": [
    {
      "message": {
        "content": "PowerShell is a powerful interactive command-line interface and scripting language included in the Windows operating system. Adversaries can use PowerShell to perform a number of actions, including discovery of information and execution of code."
      }
    }
  ]
}
```

## Prerequisites

-   Python 3.x
-   An API key for an OpenAI-compatible service (e.g., DeepSeek). The script is configured to use `https://api.deepseek.com` by default.

## Installation

1.  Clone the repository.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the pipeline from the command line.

### Command-Line Arguments

-   `--input_file` (required): Path to the input TSV file with queries.
-   `--output_file` (required): Path to save the output JSON file.
-   `--api_key` (required): Your API key for the LLM service.
-   `--bm25_cache_file` (required): Path to a file for caching BM25 results (e.g., `bm25_cache.pkl`).
-   `--model_name`: The name of the model to use for re-ranking (default: `deepseek-chat`).
-   `--bm25_top_k`: Number of documents to retrieve with BM25 (default: `100`).
-   `--rankgpt_rank_end`: The end of the slice of BM25 results to be re-ranked by RankGPT (default: `40`).
-   `--rankgpt_window_size`: The size of the sliding window for RankGPT (default: `40`).
-   `--rankgpt_step`: The step size for the sliding window (default: `20`).

### Example Command

```bash
python rag/run_ranking_pipeline.py \
    --input_file path/to/your/queries.tsv \
    --output_file results/rankings.json \
    --api_key "YOUR_API_KEY" \
    --bm25_cache_file cache/bm25.pkl \
    --model_name "deepseek-chat" \
    --bm25_top_k 100 \
    --rankgpt_rank_end 40
```

## Output Format

The script generates a JSON file specified by `--output_file`. This file contains a list of objects, one for each query in the input file.

Each object has the following structure:
- `query`: The original query string.
- `hits`: A list of ranked "hit" objects. Each hit contains the content (technique ID and summary) that was ranked.
- `true_labels`: The ground-truth technique IDs from the input file.
- `pred_labels_rankings`: A list of technique IDs in the final order produced by the RankGPT model.

### Example Output

```json
[
    {
        "query": "The malware uses PowerShell to download and execute a payload from a remote server.",
        "hits": [
            {
                "content": "T1059.001: PowerShell is a powerful interactive command-line interface..."
            },
            {
                "content": "T1105: Ingress Tool Transfer is a technique used by adversaries to download tools..."
            }
        ],
        "true_labels": [
            "T1059.001",
            "T1105"
        ],
        "pred_labels_rankings": [
            "T1059.001",
            "T1105"
        ]
    }
]
```