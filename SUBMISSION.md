# Agent Memory Leaderboard Submission

This repository implements the hosted Add/Search contract documented at:

`http://192.168.110.70:8088/api-guide`

## Docker deployment

```bash
docker build -t mi-memory-add-search .
docker run --rm \
  -p 8765:8765 \
  -v mi-memory-data:/data \
  mi-memory-add-search
```

Readiness endpoint:

```text
GET http://HOST:8765/health
```

Submission endpoints:

```text
POST http://HOST:8765/add
POST http://HOST:8765/search
```

The aliases `/v1/add` and `/v1/search` are also accepted. For a hosted integration, terminate HTTPS at the deployment platform or reverse proxy and submit the resulting public HTTPS URLs.

## Add contract

Request:

```json
{
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "messages": [
    {
      "role": "user",
      "timestamp": 1704067200000,
      "content": "raw memory text"
    }
  ],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

Successful response (`HTTP 200`):

```json
{
  "success": true,
  "request_id": "eval:<run_id>:locomo_refined:conv-0:chunk-0",
  "user_id": "eval:<run_id>:locomo:conv-0",
  "session_id": "eval:<run_id>:sample:0"
}
```

The handler persists every message before returning. A repeated `request_id` with the same payload is idempotent. Reusing it with changed messages returns `HTTP 400`.

## Search contract

Request:

```json
{
  "query": "Which answer best matches the memory?",
  "options": ["A. First answer", "B. Second answer"],
  "user_id": "eval:<run_id>:locomo:conv-0",
  "top_k": 100
}
```

Successful response (`HTTP 200`):

```json
{
  "data": [
    {
      "id": "stable-memory-id",
      "content": "remembered fact text",
      "score": 0.0325,
      "created_at": "2026-07-01T12:00:00+00:00"
    }
  ]
}
```

Results are ordered by descending relevance and never cross a `user_id` boundary. An unknown user or unmatched query returns `{"data": []}`.

## Optional endpoint authentication

The server is open by default so maintainers can deploy the public repository without a secret. To protect a hosted endpoint, set:

```bash
MIMEMORY_API_TOKEN=replace-me docker compose up --build
```

Requests must then send either:

```text
Authorization: Bearer replace-me
```

or:

```text
X-API-Key: replace-me
```

Do not enable this option unless the submission form supports passing the configured header.

## Evaluation-data retention

Each exact `user_id` is stored in a physically separate hashed directory. The server purges scopes that have not been used for 30 days at startup. It can also be run as a scheduled cleanup job:

```bash
docker run --rm -v mi-memory-data:/data mi-memory-add-search \
  mimemory-leaderboard --root /data --retention-days 30 --purge-only
```

The service does not emit request bodies or memory content into HTTP access logs.

## Local smoke test

```bash
docker compose up --build -d
python scripts/smoke_test.py --base-url http://127.0.0.1:8765
```

The smoke test exits non-zero unless Add returns the required echo fields and the added memory is immediately available through Search.

