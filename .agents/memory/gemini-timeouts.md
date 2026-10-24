---
name: Gemini request timeouts
description: Project guidance for responsive Google GenAI requests.
---

Use the Gemini async client from FastAPI async handlers. Bound both the SDK HTTP request and the overall awaited operation, and align the browser timeout to expire after the backend deadline. Map upstream request deadlines to HTTP 504 and temporary capacity/rate-limit errors to HTTP 503.

**Why:** The installed Google GenAI SDK defaults to multiple retries with exponential backoff. When the model is overloaded, this can make a request appear stuck even though the application server remains healthy.

**How to apply:** For each Gemini endpoint, keep model I/O asynchronous and set retry behavior and timeout units explicitly. Test that unrelated routes remain responsive during a model request and that timeouts reach the frontend as clear errors.

## Fallback model

Keep `gemini-3.8-flash` as the primary and use `gemini-3.6-flash` only after transient upstream failures.

**Why:** Google's model catalog lists 3.6 Flash as a stable previous-generation Flash model, providing a distinct route when 3.8 Flash is temporarily unavailable.

**How to apply:** Recheck Google's current model catalog before changing the fallback; invoke it only for timeouts and transient capacity/server errors, not permanent request or model-configuration errors.
