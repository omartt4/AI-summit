# Local PDF Assistant

## What is this application?

Local PDF Assistant is a private chatbot that lets you ask questions about your
own PDF documents.

It runs on your computer and uses local AI models through Ollama. Your
documents, questions, answers, and search index stay on your machine.

You do not need:

- OpenAI or Gemini API keys
- A paid cloud service
- An internet connection after the software and models are installed

## What can it do?

- Upload PDF files from the browser.
- Save uploaded files automatically to the local `pdfs\` folder.
- Search across multiple PDF documents.
- Answer questions using the document content.
- Show the source filename and page number for each answer.
- Create multiple chat conversations.
- Save chat history locally.
- Refresh the document library whenever PDFs are added or changed.
- Keep the previous search index if an indexing operation fails.
- Work with both exact keywords and related concepts.

## Installation

### 1. Install Python

Install Python 3.14 or newer on Windows.

### 2. Install `uv`

Install `uv` by following the instructions at:

<https://docs.astral.sh/uv/>

### 3. Install Ollama

Install Ollama from:

<https://ollama.com/>

After installation, make sure Ollama is running, then download the two required
models in PowerShell:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

These models are used locally:

- `nomic-embed-text` searches the document content.
- `qwen2.5:1.5b` writes the chatbot response.

### 4. Install the project dependencies

Open PowerShell in the project folder:

```powershell
uv sync
```

Create the local configuration file:

```powershell
Copy-Item .env.example .env
```

The `.env` file is private local configuration and must not be uploaded or
committed.

## Start the application

### First time, or after changing PDFs

You can place PDF files manually in:

```text
pdfs\
```

Or upload them from the application sidebar.

Build the document search index:

```powershell
uv run python scripts\index_pdfs.py
```

Run this command when you:

- Add a new PDF.
- Replace or edit a PDF.
- Remove a PDF.
- Want to refresh the document library.

### Open the chatbot

Start the browser application:

```powershell
uv run streamlit run app.py
```

Streamlit opens a local browser page. If it does not open automatically, copy
the local URL shown in PowerShell into your browser.

## How to use the application

1. Open the Streamlit page.
2. Upload a PDF from the sidebar, or place one in `pdfs\`.
3. Click **Index PDFs** after adding or changing documents.
4. Wait for the success message.
5. Type a question in the chat box.
6. Review the answer and expand **Sources** to see the supporting pages.
7. Click **New chat** when you want a separate conversation.

The upload control saves each selected PDF directly into the configured PDF
folder. Selecting a file does not update the search index automatically; click
**Index PDFs** after uploading.

## What happens when you ask a question?

The assistant searches the indexed PDFs, finds the most relevant passages, and
uses those passages to create an answer. It combines meaning-based search with
keyword search so it can find both related explanations and exact terms.

The answer includes source information whenever relevant passages are found.
If the PDFs do not contain enough information, the assistant is instructed to
say so instead of inventing an answer.

Text inside a PDF is treated as document content, not as instructions that can
control the assistant.

## Files and local data

| Item | Purpose |
|---|---|
| `app.py` | Browser chatbot |
| `scripts\index_pdfs.py` | Creates or refreshes the document index |
| `pdfs\` | Your PDF documents |
| `chroma_db\` | Local search index |
| `chat_history.json` | Saved conversations |
| `.env` | Private local settings |
| `.env.example` | Safe settings template |

The PDF files, search index, chat history, and `.env` are ignored by Git
because they may contain private information.

## Common problems

### Ollama connection error

Make sure Ollama is running and confirm that the models are installed:

```powershell
ollama list
```

The list should contain:

```text
nomic-embed-text
qwen2.5:1.5b
```

### No answers are found

Run the indexer before asking questions:

```powershell
uv run python scripts\index_pdfs.py
```

Then restart the browser application if necessary:

```powershell
uv run streamlit run app.py
```

### PDF text cannot be found

Some PDFs are scanned images rather than text documents. Those files need OCR
before the assistant can search their content.

### Indexing fails

Close other running copies of Streamlit or the indexer and try again. The
application is designed to preserve the previous working index when a refresh
fails.

## Limitations

- Image-only scanned PDFs need OCR.
- Answer quality depends on the local language model.
- Very large PDF collections require more memory and indexing time.
- The application answers from indexed document content; it is not a general
  internet search engine.

## Reset the local search index

To delete only the generated search index, stop the chatbot first and run:

```powershell
Remove-Item .\chroma_db -Recurse -Force
```

Your PDFs and chat history are not deleted. Run the indexer again to rebuild
the search index:

```powershell
uv run python scripts\index_pdfs.py
```
