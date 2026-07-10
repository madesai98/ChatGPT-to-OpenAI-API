# WebLLM2API

WebLLM2API exposes a local OpenAI-compatible HTTP API and sends text-generation requests through a logged-in ChatGPT browser session.

Use it when a tool, SDK, benchmark harness, or local script expects OpenAI-style endpoints but you want the actual response to come from ChatGPT in the browser. The server also logs every request and response so you can inspect what a client sent, what the server returned, and which endpoint handled the call.

## What this project provides

- POST /v1/chat/completions for OpenAI chat-completions clients.
- POST /v1/responses for OpenAI responses clients.
- GET /v1/models and GET /v1/models/{model_id} for model discovery.
- Stubbed or logged compatibility endpoints for embeddings, completions, moderations, audio, images, files, assistants, threads, batches, and vector stores.
- A catch-all /v1/{path:path} route that records unusual client calls instead of dropping them without context.
- A dashboard at /dashboard for formatted request and response inspection.
- JSONL disk logs at ./openai_mock_logs/requests.jsonl by default.
- A reusable Zendriver browser profile stored at ~/.webllm2api-zendriver-profile unless you set ZENDRIVER_PROFILE_DIR.

## Repository layout

text
.
├── app_logging.py # Shared logging setup for the server and Uvicorn
├── browser.py # Zendriver browser lifecycle and page helpers
├── browser_cli.py # Manual browser and ChatGPT command shell
├── chatgpt.py # ChatGPT website adapter and stream decoder
├── openai_bridge.py # Request translation and tool-call parsing helpers
├── openai_server.py # FastAPI app and OpenAI-compatible endpoints
├── requirements.txt # Runtime dependencies
├── run.py # CLI entrypoint for the HTTP server
├── test_openai_bridge.py # Prompt and tool-call parser tests
└── test_run.py # CLI logging tests


## Requirements

Install these before you run the server:

- Python 3.11 or newer.
- A Chromium-compatible browser that Zendriver can start.
- A ChatGPT account that can access https://chatgpt.com/.
- PowerShell, Command Prompt, Windows Terminal, macOS Terminal, or any shell that can run Python commands.

The project pins these Python packages in requirements.txt:

text
fastapi==0.115.14
uvicorn[standard]==0.35.0
python-multipart==0.0.20
zendriver==0.15.4


## Install from a local clone

Clone the repository and enter it:

bash
git clone https://github.com/<your-name>/WebLLM2API.git
cd WebLLM2API


Create a virtual environment:

bash
python -m venv .venv


Activate it on Windows PowerShell:

powershell
.\.venv\Scripts\Activate.ps1


Activate it on Windows Command Prompt:

bat
.venv\Scripts\activate.bat


Activate it on macOS or Linux:

bash
source .venv/bin/activate


Install dependencies:

bash
python -m pip install -r requirements.txt


## Sign in to ChatGPT

The server needs a persistent browser profile with a logged-in ChatGPT session. Run the manual auth command first:

bash
python browser_cli.py --headful -c "auth"


A browser window opens at ChatGPT. Sign in, complete any browser checks, then close the whole browser window. The profile saves cookies and local storage under ~/.webllm2api-zendriver-profile unless you set ZENDRIVER_PROFILE_DIR.

To store the browser profile somewhere else, set this environment variable before auth and before server startup:

Windows PowerShell:

powershell
$env:ZENDRIVER_PROFILE_DIR = "C:\Users\Mihir\.webllm2api-profile"


macOS or Linux:

bash
export ZENDRIVER_PROFILE_DIR="$HOME/.webllm2api-profile"


Use the same value every time. If you change it, the server starts with a different browser profile and ChatGPT will ask you to sign in again.

## Run the server

Start the local server:

bash
python run.py --host 127.0.0.1 --port 8000


Open the health endpoint:

text
http://127.0.0.1:8000/health


Open the dashboard:

text
http://127.0.0.1:8000/dashboard


Run with auto-reload while editing server code:

bash
python run.py --host 127.0.0.1 --port 8000 --reload


Raise log detail when you need request routing, browser, or stream diagnostics:

bash
python run.py --log-level debug


Available log levels are verbose, debug, info, warning, and error.

## Point an OpenAI client at the server

Set these values in any client that supports a custom OpenAI base URL:

bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=anything


The server redacts API keys in logged headers. It does not validate the key.

## Python client example

Install the OpenAI Python SDK in your own client environment:

bash
python -m pip install openai


Run a chat completion:

python
from openai import OpenAI

client = OpenAI(
 base_url="http://127.0.0.1:8000/v1",
 api_key="anything",
)

response = client.chat.completions.create(
 model="gpt-5-5-thinking",
 messages=[
 {"role": "user", "content": "Write one sentence about FastAPI."}
 ],
)

print(response.choices[0].message.content)


Run a responses request:

python
from openai import OpenAI

client = OpenAI(
 base_url="http://127.0.0.1:8000/v1",
 api_key="anything",
)

response = client.responses.create(
 model="gpt-5-5-thinking",
 input="Write one sentence about Zendriver.",
)

print(response.output_text)


## curl examples

Chat completions:

bash
curl http://127.0.0.1:8000/v1/chat/completions \
 -H "Authorization: Bearer anything" \
 -H "Content-Type: application/json" \
 -d '{
 "model": "gpt-5-5-thinking",
 "messages": [
 {"role": "user", "content": "Say hello in one sentence."}
 ]
 }'


Responses:

bash
curl http://127.0.0.1:8000/v1/responses \
 -H "Authorization: Bearer anything" \
 -H "Content-Type: application/json" \
 -d '{
 "model": "gpt-5-5-thinking",
 "input": "Say hello in one sentence."
 }'


Models:

bash
curl http://127.0.0.1:8000/v1/models \
 -H "Authorization: Bearer anything"


## Browser command shell

browser_cli.py gives you direct control over the persistent browser session. Use it to test ChatGPT login, send prompts, inspect pages, and capture screenshots.

Start an interactive shell:

bash
python browser_cli.py --headful


Common commands:

text
auth Open ChatGPT in a headful browser for manual login
sc Open chatgpt.com
ask TEXT Clear the composer, send TEXT, and stream the response
send [TEXT] Append optional TEXT, then send the composer
new_chat Start an empty ChatGPT conversation
open_chat URL_OR_ID Resume a ChatGPT conversation
model Print the latest rendered model slug
url Print the current page URL
screenshot PATH Save a screenshot to PATH
quit Close the shell


Run one command and exit:

bash
python browser_cli.py --headful -c "ask Write a two-word greeting."


## Configuration

Set these environment variables before starting run.py.

| Variable | Default | Effect |
| --- | --- | --- |
| HOST | 127.0.0.1 | Default host for python run.py when --host is omitted. |
| PORT | 8000 | Default port for python run.py when --port is omitted. |
| ZENDRIVER_PROFILE_DIR | ~/.webllm2api-zendriver-profile | Browser profile path that stores ChatGPT login state. |
| MOCK_OPENAI_LOG_DIR | ./openai_mock_logs | Directory for requests.jsonl. |
| MOCK_OPENAI_MAX_CAPTURE_BYTES | 10485760 | Max request or response bytes stored per event. Set 0 for unlimited capture. |
| MOCK_OPENAI_IN_MEMORY_LOG_LIMIT | 500 | Max events held by /__events and the dashboard. |
| MOCK_OPENAI_EMBEDDING_DIM | 16 | Vector size for deterministic mock embeddings. |

## Logging and inspection

The server records endpoint calls in two places:

- Disk: openai_mock_logs/requests.jsonl.
- Memory: /__events and /__logs.

Use the dashboard for quick debugging:

text
http://127.0.0.1:8000/dashboard


Clear in-memory and disk logs through the log API:

bash
curl -X DELETE http://127.0.0.1:8000/__logs


Keep logs out of public bug reports when they contain prompts, file contents, tool arguments, or private URLs. The server redacts authorization headers, but request bodies can still contain sensitive user content.

## Endpoint behavior

The main generation endpoints send text to ChatGPT through the browser session:

- POST /v1/chat/completions
- POST /v1/responses
- POST /v1/completions

Several compatibility endpoints return deterministic local data or store in-memory objects so clients can continue their setup checks:

- POST /v1/embeddings returns deterministic vectors.
- File, assistant, thread, run, batch, and vector-store endpoints use in-memory storage.
- Audio and image endpoints return API-shaped placeholder responses.
- The catch-all /v1/{path:path} returns a structured fallback and records the request.

Restarting the process clears in-memory stores. Disk logs remain under MOCK_OPENAI_LOG_DIR.

## Tool calls

When an OpenAI client supplies function tools, the server preserves the tool schema in the prompt sent to ChatGPT. If ChatGPT returns the expected tool-call JSON shape, openai_bridge.py parses it and maps it back into OpenAI-style tool calls.

The tool-call JSON shape is:

json
{
 "type": "tool_calls",
 "calls": [
 {
 "name": "read_file",
 "arguments": {
 "path": "README.md"
 }
 }
 ]
}


Windows paths need escaped backslashes inside JSON strings:

json
{
 "path": "C:\\Users\\Mihir\\Code\\WebLLM2API\\README.md"
}


The parser also repairs common client-output mistakes, including raw newlines inside JSON strings and unescaped Windows paths in tool-call payloads.

## Development checks

Run the unit tests:

bash
python -m unittest -q


Run a syntax check over the Python files:

bash
python -m compileall .


Run both before opening a pull request:

bash
python -m compileall .
python -m unittest -q


## Open-source checklist

Before publishing a repository or release package:

1. Delete local browser profiles from the repository folder if you created any there.
2. Delete openai_mock_logs/ unless you intentionally want to ship sanitized sample logs.
3. Inspect requests.jsonl for prompts, URLs, headers, file contents, and tool arguments.
4. Keep .venv/, __pycache__/, .pytest_cache/, and build output out of git.
5. Add a license file that matches how you want other developers to use the project.
6. Run python -m unittest -q from a clean virtual environment.

## Troubleshooting

### Zendriver is not installed

Install dependencies again inside the active virtual environment:

bash
python -m pip install -r requirements.txt


### The server opens ChatGPT but stays signed out

Run auth with the same ZENDRIVER_PROFILE_DIR that the server uses:

bash
python browser_cli.py --headful -c "auth"


Close the whole browser window after login. A tab close may not flush profile state on every platform.

### Another process is using the browser profile

Stop other python run.py and browser_cli.py processes. Zendriver needs exclusive access to the profile directory.

### A client receives a browser error

Open the dashboard and inspect the failed request. Then run a direct shell test:

bash
python browser_cli.py --headful -c "ask Say hello in one short sentence."


If the shell test fails, sign in again with auth and confirm ChatGPT can send a normal message in the visible browser.

### Logs grow too large

Move logs to a temporary folder:

bash
MOCK_OPENAI_LOG_DIR=/tmp/webllm2api-logs python run.py --port 8000


Limit captured body size:

bash
MOCK_OPENAI_MAX_CAPTURE_BYTES=200000 python run.py --port 8000


### A benchmark calls an endpoint that is not implemented

Check openai_mock_logs/requests.jsonl or /dashboard. The catch-all route records the exact path, method, headers, and body so you can add a focused endpoint in openai_server.py.

## Security notes

Run this server on 127.0.0.1 unless you have added authentication, network controls, and log retention rules. The default server accepts any API key and allows CORS from any origin.

Treat the browser profile as a secret. It stores a logged-in ChatGPT session. Do not commit it, upload it, or share it.

Treat logs as private data. Request bodies can contain source code, prompts, file paths, private documents, tool outputs, and conversation URLs.