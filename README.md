# BS Translator Backend

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)](https://img.shields.io/github/license/DCC-BS/bs-translator-backend)

BS Translator Backend is a powerful Python FastAPI service that provides advanced text translation and document conversion capabilities. This backend service enables high-quality translation of text and documents using state-of-the-art AI models with customizable translation parameters.

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
- **LLM Orchestration**: [DSPy](https://github.com/stanfordnlp/dspy) with OpenAI/vLLM-compatible endpoints
- **Document Processing**: Docling for document conversion
- **Containerization**: Docker and Docker Compose

## Setup

### Prerequisites

- Python 3.13
- uv package manager
- Docker and Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA support (for LLM services)

### Environment Configuration

Create a `.env` file in the project root with the required environment variables:

```env
# Hugging Face Configuration (optional)
HF_AUTH_TOKEN=your_hugging_face_token_here
HUGGING_FACE_CACHE_DIR=~/.cache/huggingface

# LLM Service Configuration (vLLM/OpenAI-compatible)
LLM_API_PORT=8001
OPENAI_API_BASE_URL=http://localhost:${LLM_API_PORT}/v1
OPENAI_API_KEY='none'
LLM_MODEL='ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g'

# Optional DSPy optimizer (used by optimize_llm.py)
OPTIMIZER_API_KEY=
OPTIMIZER_MODEL=
OPTIMIZER_API_BASE_URL=

# Client Configuration
CLIENT_PORT=3000
CLIENT_URL=http://localhost:${CLIENT_PORT}

# Service URLs and auth
HMAC_SECRET='your_hmac_secret' # generate with openssl rand 32 | base64
DOCLING_URL='http://localhost:8004/v1'
WHISPER_URL='http://localhost:50001/v1'
```

> **Note:** The optimizer values are only needed when running DSPy optimization scripts. The `HF_AUTH_TOKEN` is required for Hugging Face API access and model downloads; you can create a token [here](https://huggingface.co/settings/tokens).

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
uv run fastapi run ./src/bs_translator_backend/app.py
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
- **Backend API**: FastAPI application for translation services

### Using Dockerfile Only

```bash
# Build the Docker image
docker build -t bs-translator-backend .

# Run the container
docker run --rm --env-file .env -p 8000:8000 bs-translator-backend
```

## Testing & Development Tools

Run tests with pytest:

```bash
# Run tests
make test

# Run tests with pytest directly
uv run python -m pytest --doctest-modules
```

### DSPy Translation Optimization

- Prepare sample and custom datasets (writes CSVs under `src/bs_translator_backend/data/`):
  - `uv run prepare-dataset
- Optimize the translation module with DSPy (requires `OPTIMIZER_*` env vars):
  - `uv run optimize-translation
  - Outputs an updated `translation_module.pkl` used by the API for streaming translations.

## API Endpoints

### Translation

- **GET `/translation/languages`**: Get list of supported languages
- **POST `/translation/text`**: Translate text with customizable parameters

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
├── data/                       # Dataset files
├── models/                     # Data models and schemas
│   ├── app_config.py          # Application configuration
│   ├── conversion_result.py   # Document conversion models
│   ├── language.py           # Language definitions
│   ├── translation_config.py  # Translation configuration
│   └── translation_input.py   # Translation input models
├── routers/                    # API endpoint definitions
│   ├── convert_route.py       # Document conversion endpoints
│   └── translation_route.py   # Translation endpoints
├── scripts/                    # Offline utilities
│   ├── optimize_llm.py        # DSPy optimization entry point
│   └── prepare_dataset.py     # Dataset preparation helper
├── services/                   # Business logic services
│   ├── document_conversion_service.py  # Document processing
│   ├── dspy_config/           # DSPy translation program and assets
│   │   ├── dataset_loader.py  # Dataset loading and sampling
│   │   ├── translation_program.py # DSPy translation module
│   │   ├── translation_module.pkl # Optimized module artifact
│   │   └── small_translation_module.pkl # Lightweight module
│   ├── text_chunk_service.py  # Text chunking utilities
│   ├── transcription_service.py # Whisper transcription integration
│   ├── translation_service.py # Translation logic
│   └── usage_tracking_service.py # Request tracking
└── utils/                      # Utility functions and helpers
    ├── cancelation.py         # Cancellation utilities
    ├── image_overlay.py       # Image overlay helpers
    ├── language_detection.py  # Language detection utilities
    ├── load_env.py            # Environment loading
    └── logger.py              # Logging configuration
```



## License

[MIT](LICENSE) © Data Competence Center Basel-Stadt

---

<a href="https://www.bs.ch/schwerpunkte/daten/databs/schwerpunkte/datenwissenschaften-und-ki"><img src="https://github.com/DCC-BS/.github/blob/main/_imgs/databs_log.png?raw=true" alt="DCC Logo" width="200" /></a>

**Datenwissenschaften und KI**
Developed with ❤️ by DCC - Data Competence Center
