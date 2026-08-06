# NexusFlow

NexusFlow is a Retrieval-Augmented Generation (RAG) platform: drop documents in, they get chunked and embedded into a vector store, and a chat UI answers questions grounded strictly in that content — with cited source passages, and an explicit "I don't know" when the documents don't contain the answer.

It's built as an architecture demo first: a document processing pipeline, a vector store, an LLM-serving layer, and a UI, wired together as independently swappable pieces rather than a single script. See `System_Diagrams/BlockDiagram.drawio` for the full architecture.

## How it works

```
docs/*.pdf → ingest.py → chunk → embed → Pinecone (vector index)
                                                │
                                                ▼
Streamlit UI (app.py) → BentoML service (service.py) → retrieve top-k chunks
                                                       → prompt an LLM with retrieved context
                                                       → grounded answer + sources
```

- **Ingestion** (`ingest.py`) — extracts text from PDFs, splits into overlapping chunks (`langchain-text-splitters`), embeds each chunk with a `sentence-transformers` model, and upserts into a Pinecone index with source/chunk metadata attached.
- **Serving** (`service.py`) — a BentoML service that embeds an incoming query, retrieves the top-k most relevant chunks from Pinecone, and prompts an LLM to answer *only* from that retrieved context. Currently runs against a local Ollama model; there's a commented-out AWS Bedrock path in the same method, laid in place for the AWS deployment described in the Roadmap below.
- **UI** (`app.py`) — a Streamlit chat interface that calls the BentoML service over HTTP.

## Setup

1. Create a venv and install dependencies:
   ```
   python -m venv venv
   ./venv/Scripts/pip install -r requirements.txt
   ```
2. Create a `.env` file (gitignored) with:
   ```
   EMBEDDING_MODEL_NAME=   # a sentence-transformers model id, e.g. nomic-ai/nomic-embed-text-v1.5
   PINECONE_API_KEY=
   PINECONE_INDEX_NAME=    # the index NAME, not its host subdomain
   OLLAMA_API_URL=         # e.g. http://localhost:11434
   OLLAMA_MODEL_NAME=      # whatever you've `ollama pull`ed
   ```
   The Pinecone index's vector dimension must match your embedding model's output dimension — check both before ingesting.
3. Make sure [Ollama](https://ollama.com) is running locally with your chosen model pulled: `ollama pull <model>`.
4. Drop PDFs into `docs/`, then run ingestion:
   ```
   ./venv/Scripts/python ingest.py
   ```
5. Start the service and UI in separate terminals:
   ```
   ./venv/Scripts/python -m bentoml serve service:NexusFlowService --port 3000
   ./venv/Scripts/streamlit run app.py
   ```

## Roadmap

The current pipeline runs entirely locally by design, to validate the architecture end-to-end before layering on cloud infrastructure. Planned next:

- **AWS-based batch processing backend** — event-driven ingestion (S3 upload → Lambda/Step Functions → embed → Pinecone) in place of the current manual `python ingest.py` run, plus distributed/batched embedding.
- **AWS Bedrock integration** — swap the local Ollama call for the already-stubbed Bedrock path in `service.py`, for a fully cloud-hosted inference story.
- **Containerization** — Dockerfile for the BentoML service.
- **Experiment tracking** — track retrieval/generation quality across embedding models and prompt versions.
- **Tests** — unit coverage for the chunking/extraction pipeline, integration coverage for the retrieve→generate loop.

CI/CD is intentionally out of scope for now.
