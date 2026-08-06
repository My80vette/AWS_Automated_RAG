# NexusFlow — TODO

Last audited: 2026-08-05. Goal for this pass: **Loom-ready demo** — drop a doc in, watch it get processed, ask it a question and get a grounded answer. Phases 0-3 (the pipeline itself) are done and verified end-to-end as of this session. What's left is UI polish and the recording itself. Productionization work (Docker, CI/CD, Bedrock, AWS event-driven pipeline) is parked in the Backlog.

Status key: `[ ]` not started · `[~]` partially done / needs verification · `[x]` done

---

## Phase 0 — Environment & secrets ✅ done

- [x] venv created, `pip install -r requirements.txt` (plus `einops`, required by the embedding model — not in `requirements.txt`, worth adding).
- [x] `.env` populated. Real bug found and fixed along the way: `PINECONE_INDEX_NAME` had been set to the *host subdomain* (`aws-rag-custom-1fscdsp`) instead of the actual index name (`aws-rag-custom`) — `pc.describe_index()` was 404ing until corrected.
- [x] Embedding model resolved to `nomic-ai/nomic-embed-text-v1.5` (768-dim) to match the existing Pinecone index's dimension — confirmed by loading the model and checking output shape before running any real ingestion.
- [x] Ollama confirmed running (`localhost:11434`), `llama3.1:8b` already pulled and set as `OLLAMA_MODEL_NAME`.
- [ ] **`config.py` is still dead code** — `ingest.py`/`service.py` read from `.env` via `os.environ`, never from `config.py`, and its values are still placeholders (`"Placeholder"`). Wire it in or delete it — currently just misleading to anyone reading the repo.

## Phase 1 — Demo story (not blocking, your call)

- [ ] Naming/branding mismatch (K8s title vs. CS-paper docs) — nitpick, not a technical blocker, explicitly deprioritized. Handle whenever.
- [ ] Still worth doing regardless of theme: trim `docs/` down to 2-4 files for the demo. Full ingestion of all 5 (including a 6MB neural networks book, 1078 of the 1479 total chunks) took a few minutes on CPU since `ingest.py` embeds one chunk at a time — fine for a one-time setup, but slower than it needs to be to "watch it process" live on camera.

## Phase 2 — Ingestion ✅ done

- [x] Real run against all 5 PDFs in `docs/`: 1479 chunks total, all upserted to Pinecone. Verified independently by listing vector IDs back out of Pinecone and parsing the `{filename}_{uuid8}` pattern — counts matched exactly (129 + 1078 + 90 + 99 + 83).
- [x] Found ~1482 **stale vectors** already in the index from an old test run, with no recoverable record of which embedding model produced them (checked `git log -p -- config.py` across all commits — always `"Placeholder"`, real value only ever lived in the gitignored `.env`). Wiped (`index.delete(delete_all=True)`) rather than risk silently mixing incompatible vector spaces, then re-ingested clean.
- [ ] Add per-file error handling — one bad/corrupt/scanned PDF (`extract_text_from_pdf` returns empty string for image-only PDFs) will still throw or silently produce zero chunks and kill the whole batch loop. Didn't hit this with the current 5 files, but untested against a bad input.
- [ ] Add visible progress output suited for screen recording — current `print()`s are functional (confirmed readable in this session's run), consider `tqdm` if you want it tighter for camera.
- [ ] **Still not idempotent** — every run generates new random `uuid` vector IDs, so running `ingest.py` twice on the same file duplicates every chunk. Confirmed this is a real risk, not theoretical (it's exactly what the stale-vector situation was). Derive the id from `filename + chunk_index` (or a content hash) instead of `uuid.uuid4()`.
- [ ] Decide if ingestion should be triggerable **from the UI** (upload → ingest → ready) for the "drop in docs" beat, or if the Loom narrative is "run `ingest.py` in a terminal, watch the output, then switch to the app." Either is fine — just decide since it changes Phase 4 scope.

## Phase 3 — BentoML service ✅ done

- [x] `bentoml serve service:NexusFlowService --port 3000` boots clean — embedding model load, Pinecone connection, and Ollama connection in `__init__` all succeeded.
- [x] Smoke-tested `/answer_question` directly via curl (bypassing the UI, so ingestion/service bugs wouldn't get conflated):
  - In-scope question ("What is a perfect hash function?") → correct grounded answer with real quoted source chunks from the perfect-hashing paper.
  - Out-of-scope question ("What is the capital of France?") → correctly returned "I don't know" instead of hallucinating. Guardrail works as designed.
- [ ] Add basic error handling: what happens today if Pinecone returns zero matches, or Ollama is unreachable? Not yet tested. `ollama_response.raise_for_status()` will throw an unhandled exception straight through the API — wrap it and return a clean error message instead of a 500.
- [ ] Consider trimming the prompt template — functional but a bit redundant; not urgent.
- [ ] (Optional, nice for a live demo) Stream the Ollama response (`stream: True` + BentoML streaming API) so the answer appears incrementally instead of one blocking POST.

## Phase 4 — Streamlit UI polish for the recording

- [ ] Sidebar is still an empty placeholder (`# Add your options here as you develop`) — at minimum add:
  - A "sources" expander under each answer — `service.py` already returns `sources` in the response payload (confirmed working via curl this session) but `app.py` currently discards it (`response_data.get("answer", ...)` only, never reads `["sources"]`). Easiest, highest-impact fix left — displaying retrieved chunks/citations is the whole point of a RAG demo.
  - top_k slider or model indicator, if time allows — not required for the demo.
- [ ] Handle the BentoML-unreachable case gracefully — right now a connection error renders as `Error: HTTPConnectionError(...)` inline in the chat. Catch it and show a friendlier "service not running" message.
- [ ] Minor: the assistant chat role is set to the literal string `"Kubernetes Manager"` — Streamlit's `st.chat_message` gives nicer default avatars for `"assistant"`/`"ai"`. Cosmetic.
- [ ] If Phase 2's UI-vs-terminal ingestion question lands on "in-UI": add a file uploader + a "processing…" state that calls into the ingestion pipeline and reports progress before the chat unlocks.

## Phase 5 — Loom recording checklist (once Phase 4 is done)

- [x] Clean index — done this session, no stale/duplicate data.
- [ ] Confirm cold-start timing — embedding model + Ollama model load can take a while on first call (this session's first `bentoml serve` startup wasn't timed precisely; budget for it or narrate through the wait).
- [ ] Script the beats: drop file(s) in `docs/` → run ingestion (show the terminal output) → switch to the Streamlit app → ask 2-3 questions that clearly demonstrate grounded answers + cited sources → ask one question the docs *can't* answer to show the "I don't know" guardrail firing (already confirmed working via curl — just needs to happen in the UI too).

---

## Backlog (post-demo / productionization — not needed for the Loom)

**Next up after this checkpoint:** architecting an AWS backend for real batch processing (event-driven ingestion, distributed embedding) — this is the planned next conversation, not yet scoped.

The README previously promised more than the code delivered; it's being rewritten alongside this TODO to reflect actual current state plus an honest roadmap. Backlog items to build toward:

- [ ] AWS Bedrock path — fully commented out in `service.py`; currently 100% local (Ollama). Needed for the "AWS-native" story beyond the demo.
- [ ] Event-driven ingestion (S3 upload trigger → Lambda/Step Functions → embed → Pinecone) — doesn't exist yet; current ingestion is a manual local script. **This is the next major piece of work.**
- [ ] Distributed embedding — currently single-process, single-machine, and not even batched within a single run (`ingest.py` calls `embedding_model.encode()` once per chunk in a loop).
- [ ] Experiment tracking — none present (no MLflow/W&B, etc.).
- [ ] Dockerfile / containerization — no Dockerfile exists yet.
- [ ] CI/CD — no `.github/workflows`. **Explicitly on hold — GitHub Actions requires Nick's direct sign-off before any workflow file is added** (repo/billing implications), not just "not built yet."
- [ ] Tests — zero tests in the repo. At minimum, unit-test `chunk_text` and `extract_text_from_pdf`, and an integration test for `answer_question` against a mocked Pinecone/Ollama.
- [ ] Revisit `System_Diagrams/BlockDiagram.drawio` once the AWS backend work lands — confirm it still matches reality.
