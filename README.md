# Document search: RAG proof of concept

A small, working retrieval-augmented search over company documents. Upload Word, Excel, PDF or text files, then ask questions in Arabic or English. Every answer carries exact quotes from the source, with the document and location.

This proof of concept shows the architecture and the code structure. It does not include role-based access control (only a single access level per document), the other EMS modules, or the production database.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

```bash
# upload a sample file
curl -F "file=@samples/booth_contracts.txt" http://127.0.0.1:8000/documents

# ask a question
curl -H "Content-Type: application/json" \
     -d '{"question": "What are the payment terms for booth 12?"}' \
     http://127.0.0.1:8000/query
```

Run the tests and the evaluation:

```bash
pytest -q
python eval/run_eval.py
```

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Status, and which embedder, reranker and answer mode are active |
| `POST /documents` | Upload a file. Returns the document ID and passage count. Uploading the same bytes twice returns the same document. |
| `GET /documents/{id}` | Document details |
| `GET /documents/{id}/download` | The original file, byte for byte |
| `POST /query` | Ask a question. Returns `outcome` (`B` answer, `E` not found, `OT` refused), the answer, quotes, and a message where relevant |

## How it works

**Upload.** The file is read into sections, each with a location (page, sheet and row, or heading). Sections are split into overlapping passages of about 300 words, and each passage keeps exact character offsets into its section. The passages are embedded and stored in the index.

**Question.** Payment requests are refused with the SRS message. Otherwise the question is searched two ways: keyword search (BM25, with Arabic normalisation) and vector search. The two lists are merged with reciprocal rank fusion. A reranker scores the top candidates against the question, and passages below the relevance threshold are dropped. If nothing is left, the not-found message is returned.

**Answer.** The language model, if configured, writes the answer from the selected passages only. The quotes are never written by the model: each quote is the most relevant sentence of a stored passage, and it is checked to be an exact substring of that passage before it is returned.

## What is real and what is a placeholder

| Part | Default in this PoC | Production choice |
|---|---|---|
| Embeddings | Hashing embedder: matches shared words and character fragments, no meaning | BAAI/bge-m3 (`EMBEDDER=bge-m3`) |
| Reranker | Lexical scorer with number matching | BAAI/bge-reranker-v2-m3 (`RERANKER=bge`) |
| Answer wording | Extractive: the best-matching quote, no model call | Claude API with a real key in `ANTHROPIC_API_KEY` |
| Index | In memory | pgvector or OpenSearch behind the same index methods |
| Access control | One access level per document, applied inside the search | Full role-based filtering |

With the default placeholders, search finds documents by shared words and the answers are quotes rather than written summaries. This is enough to demonstrate the flow and the verification. Real semantic quality needs the models in `requirements-ml.txt`. The relevance threshold (default 0.5) is tuned for the lexical reranker, so it must be recalibrated when the cross-encoder is enabled.

Known limitations:
- Cross-lingual search (an English question finding an Arabic passage) needs bge-m3. The placeholder embedder does not do it.
- The in-memory index scans every passage per query. It is fine for a demo, not for millions of passages.
- Arabic payment requests are not yet recognised by the refusal rule.
- Hidden spreadsheet sheets are skipped. Scanned PDFs and password-protected files are not read.
- Grading is simplified: there is no separate handling for multiple documents (output D) or ambiguous questions (output C).

## Scaling path

- The API is stateless, so it can run as several containers behind a load balancer.
- Ingestion can move to a queue and worker processes, separate from query serving.
- The index interface (`add_document`, `keyword`, `vector`) can be implemented by pgvector or OpenSearch without changing the services.
- Embedding and reranking can run on a GPU host when volume grows.

## Project layout

```
app/
  main.py          HTTP routes only
  container.py     chooses the concrete adapters from settings
  config.py        environment settings
  schemas.py       data shapes
  text.py          Arabic normalisation, SRS messages
  chunking.py      overlapping passages with exact offsets
  parsing.py       reads PDF, Word, Excel and text
  adapters/        embedder, index, reranker, answer model: one interface each
  services/        ingest and query logic
tests/             API and chunking tests
eval/              golden set and evaluation script
samples/           sample documents (invented)
```
