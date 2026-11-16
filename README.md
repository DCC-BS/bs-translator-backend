# BS Translator Backend

BS Translator Backend is a powerful Python FastAPI service that provides advanced text translation and document conversion capabilities. This backend service enables high-quality translation of text and documents using state-of-the-art AI models with customizable translation parameters.

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)

---

<p align="center">
  <a href="https://dcc-bs.github.io/documentation/">DCC Documentation & Guidelines</a> | <a href="https://www.bs.ch/daten/databs/dcc">DCC Website</a>
</p>

---

## Features

- **Text Translation**: High-quality text translation with customizable parameters
- **Document Conversion**: Convert various document formats (PDF, DOCX) to markdown with image extraction
- **Language Detection**: Automatic source language detection
- **Customizable Translation**: Configure tone, domain, glossary, and context for translations
- **Streaming Response**: Real-time translation output for improved user experience
- **Multi-format Support**: Handle text, PDF, and DOCX documents
- **Image Extraction**: Extract and encode images from documents during conversion

## Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) with Python 3.13
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Dependency Injection**: Dependency-Injector
- **LLM Orchestration**: [Pydantic AI](https://ai.pydantic.dev/) with OpenAI/vLLM-compatible endpoints
- **Document Processing**: Docling for document conversion
- **Containerization**: Docker and Docker Compose

## Setup

### Prerequisites

- Python 3.13
- uv package manager
- Docker and Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA support (for LLM services)

### Environment Configuration

This project uses [varlock](https://varlock.dev) for environment variable management. The `.env.schema` file defines all required variables with types, defaults, and validation.

For local development, create a `.env` file in the project root:

```env
# Hugging Face Configuration (optional)
HF_TOKEN=your_hugging_face_token_here
HF_HOME=~/.cache/huggingface

# LLM Service Configuration (vLLM/OpenAI-compatible)
LLM_API_PORT=8001
LLM_URL=http://localhost:${LLM_API_PORT}/v1
LLM_API_KEY='none'
LLM_REASONING=false
LLM_MODEL='Qwen/Qwen3-32B-AWQ'

# Client Configuration
CLIENT_PORT=3000
CLIENT_URL=http://localhost:${CLIENT_PORT}

# Service URLs and auth
HMAC_SECRET='your_hmac_secret' # generate with openssl rand 32 | base64
DOCLING_URL='http://localhost:8004/v1'
WHISPER_URL='http://localhost:50001/v1'

# Uploads and photo translation (defaults shown)
MAX_UPLOAD_SIZE_MB=50
DOCLING_IMAGE_OCR_PRESET='glm-ocr-remote'
DOCLING_IMAGE_LAYOUT_PRESET='ppdoclayout-v3'
IMAGE_TRANSLATION_CONCURRENCY=4
```

> **Note:** The `HF_TOKEN` is required for Hugging Face API access and model downloads; you can create a token [here](https://huggingface.co/settings/tokens). Set `LLM_REASONING=true` to enable extended reasoning/thinking in the LLM responses.

### Install Dependencies

Install dependencies using uv:

```bash
make install
```

This will:
- Create a virtual environment using uv
- Install all dependencies
- Install pre-commit hooks

## Development

### Start the Development Server

```bash
make dev
```

### Code Quality Tools

Run code quality checks:

```bash
# Run all quality checks
make check

# Format code with ruff
uv run ruff format .

# Run linting
uv run ruff check .

# Run type checking
uv run basedpyright
```

## Production

Run the production server:

```bash
uv run uvicorn bs_translator_backend.app:app --no-access-log
```

## Docker Deployment

The application includes a Dockerfile and Docker Compose configuration for easy deployment with LLM services:

### Using Docker Compose

```bash
# Start all services with Docker Compose
docker compose up -d

# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f
```

The Docker Compose setup includes:
- **vLLM Service**: Serves the configured model with GPU acceleration
- **docling-serve**: Document and photo conversion (`ghcr.io/dcc-bs/dcc-docling-serve-cu130`), with the PP-DocLayout-V3 layout preset and GLM-OCR remote OCR
- **vLLM GLM-OCR**: Serves `zai-org/GLM-OCR` for docling-serve's `glm-ocr-remote` OCR preset

### Using Dockerfile Only

```bash
# Build the Docker image
docker build -t bs-translator-backend .

# Run the container (listens on $PORT, default 8000)
docker run --rm --env-file .env -p 8000:8000 bs-translator-backend
```

The image never reads a local `.env` (it is excluded from the build context) and contains
the full fastText language-identification model (`lid.176.bin`) at `$FTLANG_CACHE`, so
language detection never downloads anything at runtime. Outside Docker the app uses
`$FTLANG_CACHE/lid.176.bin` when present and otherwise falls back to the smaller model
bundled with fast-langdetect.

## Testing & Development Tools

Run tests with pytest (requires a `.env` file):

```bash
# Run tests
make test

# Run tests with pytest directly
uv run python -m pytest --doctest-modules
```

## API Endpoints

### Translation

- **GET `/translation/languages`**: Get list of supported languages
- **POST `/translation/text`**: Translate text with customizable parameters. Streams plain text; if the translation fails after the first chunk was sent, the stream ends with the character U+001E followed by `{"errorId": "...", "status": <int>}`
- **POST `/translation/detect-language`**: Detect language of a given text with confidence score (`auto` when the confidence is below 0.5)

Errors are JSON `{"errorId": "...", "status": <int>, "debugMessage": "..."}`; request validation errors (422, `invalid_request`) also carry FastAPI's `detail`.

### Document Conversion

- **POST `/convert/doc`**: Convert documents (PDF, DOCX) to markdown with image extraction

### Translation Configuration

The translation service supports the following customizable parameters:

- **target_language**: Target language for translation
- **source_language**: Source language (auto-detected if not specified)
- **domain**: Domain or subject area for translation
- **tone**: Translation tone (formal, informal, technical, neutral)
- **glossary**: Custom glossary or terminology
- **context**: Additional context for translation

## Project Architecture

```
src/bs_translator_backend/
├── app.py                      # FastAPI application entry point
├── container.py                # Dependency injection container
├── agents/                     # AI agent definitions
│   └── translation_agent.py   # Pydantic AI translation agent
├── data/                       # Dataset files
├── models/                     # Data models and schemas
│   ├── conversion_result.py   # Document conversion models
│   ├── language.py            # Language definitions
│   └── translation.py         # Translation configuration and input models
├── routers/                    # API endpoint definitions
│   ├── convert_route.py       # Document conversion endpoints
│   └── translation_route.py   # Translation endpoints
├── services/                   # Business logic services
│   ├── document_conversion_service.py  # Document processing
│   ├── text_chunk_service.py  # Text chunking utilities
│   ├── transcription_service.py # Whisper transcription integration
│   ├── translation_service.py # Translation logic
│   └── usage_tracking_service.py # Request tracking
└── utils/                      # Utility functions and helpers
    ├── app_config.py          # Application configuration
    ├── cancelation.py         # Cancellation utilities
    ├── image_overlay.py       # Image overlay helpers
    └── language_detection.py  # Language detection utilities
```



## License

[MIT](LICENSE) © Data Competence Center Basel-Stadt

---

<a href="https://www.bs.ch/schwerpunkte/daten/databs/schwerpunkte/datenwissenschaften-und-ki"><img src="https://github.com/DCC-BS/.github/blob/main/_imgs/databs_log.png?raw=true" alt="DCC Logo" width="200" /></a>

**Datenwissenschaften und KI**
Developed with ❤️ by DCC - Data Competence Center
