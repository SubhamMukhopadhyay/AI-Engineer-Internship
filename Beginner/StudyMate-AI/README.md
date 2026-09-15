# 📚 StudyMate AI (Beginner Project)

A real, working AI-powered student utility app built with **Streamlit**.
It calls a live LLM API for every response — there are **no hardcoded or
canned outputs**.

## Features

- **Summarize Notes** — condenses pasted notes into revision bullets
- **Generate Quiz** — 5 MCQs + 2 short-answer questions with an answer key
- **Improve My Answer** — rewrites a draft exam/homework answer with a changelog
- **Explain Concept** — definition + analogy + worked example
- **Build Study Plan** — turns topics/notes into a session-by-session plan
- Copy/download result as `.txt`
- Clear/reset button
- Loading spinner during generation
- Empty-input validation, too-short/too-long input handling
- Explicit, human-readable error messages for API/config failures

## Architecture

```
app.py          Streamlit UI: layout, state, validation, wiring
prompts.py      Builds (system_prompt, user_prompt) per feature
llm_client.py   Provider-agnostic OpenAI-compatible HTTP client
```

Data flow: `user input → prompts.py builds a prompt → llm_client.py calls
the API → app.py renders the result or a clear error`.

## Installation

```bash
cd Beginner/StudyMate-AI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real key
```

## Environment variables

| Variable       | Description                                         |
|----------------|------------------------------------------------------|
| `LLM_API_KEY`  | Secret key for your LLM provider                     |
| `LLM_BASE_URL` | Base URL of an OpenAI-compatible `/chat/completions` API |
| `LLM_MODEL`    | Model name, e.g. `gpt-4o-mini`, `llama-3.1-8b-instant` |

Works with any OpenAI-compatible provider (OpenAI, Groq, OpenRouter, a local
vLLM/Ollama proxy, etc.) — just change the three variables above.

## Run

```bash
streamlit run app.py
```

## Prompt design

Every feature has a **fixed system prompt** (defines the assistant's role
and hard constraints, e.g. "never invent facts not in the notes") and a
**dynamic user prompt** (injects the student's pasted content with explicit
formatting rules). Separating these two keeps outputs consistent and
auditable — see `prompts.py` for the exact text sent for each feature.

## Error handling

- **Empty input** → inline warning, no API call made
- **Too short (<10 chars) / too long (>20,000 chars)** → inline warning
- **Missing `.env` configuration** → explicit configuration error, no request sent
- **Network/timeout/connection errors** → caught and shown with the specific cause
- **HTTP 401 / 404 / 429 / 5xx** → mapped to a specific, actionable message
- **Empty or malformed API response** → treated as a failure, never silently
  replaced with fake text

## Example usage

1. Paste a paragraph of biology notes.
2. Select **Generate Quiz**.
3. Click **Generate** → the app sends the notes to your configured LLM and
   displays 5 MCQs + 2 short-answer questions with an answer key.
4. Click **Copy / Download result as .txt** to save it.

## Limitations

- Requires a valid, funded LLM API key — the app does not run without one.
- No persistence between sessions (by design, for a lightweight beginner tool).
