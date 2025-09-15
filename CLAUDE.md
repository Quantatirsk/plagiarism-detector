# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI-based plagiarism detection system using vector embeddings, designed with Linus Torvalds' principle: "Talk is cheap. Show me the code."

The system uses:

- **OpenAI-compatible API** for text embeddings (configured for Qwen3-Embedding-8B model via vect.one)
- **Milvus vector database** with dual-mode support (local file-based for development, server for production)
- **FastAPI** for REST API endpoints

## Key Development Commands

### Initial Setup

```bash
# Initialize project structure
python scripts/init_project.py

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API credentials
```

### Running the Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs
```

### Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_detection.py -v

# Run with coverage
pytest --cov=app tests/
```

## Architecture & Design Principles

### Core Philosophy

1. **简单性 (Simplicity)**: Keep it simple, avoid complexity
2. **实用性 (Practicality)**: Make it work first, optimize later
3. **清晰性 (Clarity)**: Code is documentation, naming is commentary

### Milvus Storage Modes

The system supports dual storage modes controlled by `MILVUS_MODE` environment variable:

- **Local Mode** (`MILVUS_MODE=local`): Uses `MilvusClient("milvus_demo.db")` for development - no server required
- **Server Mode** (`MILVUS_MODE=server`): Connects to production Milvus server

Storage service automatically selects appropriate connection method based on mode in `app/services/storage.py`.

### Service Architecture

The application follows a layered architecture:

```
API Layer (FastAPI endpoints) → Service Layer (Business Logic) → Repository Layer (Storage/Cache)
                              ↓
                        Models (Pydantic schemas)
```

Key services work together:

- **EmbeddingService** → Calls OpenAI-compatible API for text embeddings
- **TextProcessor** → Splits text into paragraphs/sentences for analysis
- **DetectionService** → Orchestrates embedding + storage + similarity search
- **MilvusStorage** → Handles vector storage with mode-based routing

### Detection Flow

1. **Text Processing**: Input text split into paragraphs (primary) and sentences (optional)
2. **Embedding Generation**: Batch API calls to embedding service with retry logic
3. **Similarity Search**: Vector search in Milvus with cosine similarity
4. **Two-Level Detection**:
   - Paragraph-level: Fast, threshold 0.75
   - Sentence-level: Detailed (modes: detailed/comprehensive), threshold 0.80

### Critical Configuration

Key environment variables in `.env`:

- `OPENAI_API_KEY`: Required for embedding API access
- `OPENAI_BASE_URL`: Custom endpoint (currently using vect.one)
- `OPENAI_MODEL`: Embedding model (Qwen3-Embedding-8B)
- `OPENAI_DIMENSIONS`: Vector dimensions (4096 for current model)
- `MILVUS_MODE`: Storage mode (local/server)

### Development Workflow

The project follows incremental development approach outlined in `docs/todo.json`:

1. **Phase 1-3**: Environment setup, configuration, models
2. **Phase 4-5**: Core services (text, embedding, storage, detection)
3. **Phase 6-7**: API implementation and optimization
4. **Phase 8-9**: Testing and validation
5. **Phase 10**: Production preparation

Current implementation status tracked in todo list - use TodoWrite to update progress.

### Important Implementation Notes

- **Embedding Model**: Using Qwen3-Embedding-8B via vect.one API (it is OpenAI API compatible)
- **Vector Dimensions**: 4096
- **Batch Size**: Reduced to 20 for current model limitations
- **Local Development**: Always start with `MILVUS_MODE=local` to avoid server dependencies
- **Schema Management**: MilvusClient handles schema automatically in local mode, explicit schema required for server mode

## Common Development Tasks

### Switch Between Storage Modes

```python
# In .env file
MILVUS_MODE="local"   # For development
MILVUS_MODE="server"  # For production
```

### Clear Local Database

```bash
rm milvus_demo.db  # Remove local Milvus database file
```

### Test Embedding Service

```python
from app.services.embedding import EmbeddingService
import asyncio

async def test():
    service = EmbeddingService()
    vector = await service.embed_text("test")
    print(f"Vector dimensions: {len(vector)}")  # Should be 4096

asyncio.run(test())
```

## Current Development Focus

Based on `docs/comprehensive-implementation.md`, the project follows a structured implementation plan with emphasis on:

1. Getting minimal viable product running with local Milvus
2. Implementing core detection logic with two-level similarity analysis
3. Optimizing batch processing and API calls
4. Adding monitoring and error handling progressively

Refer to `docs/development-roadmap.md` for day-by-day implementation guide.
