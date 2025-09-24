# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance plagiarism detection system using vector embeddings, with support for Chinese/English intelligent text segmentation and real-time interactive comparison.

Key Technologies:
- **Frontend**: React 19.1 + TypeScript 5.7 + Vite + Tailwind CSS + Radix UI
- **Backend**: FastAPI + spaCy 3.7+ + Milvus 2.5+ + OpenAI-compatible embeddings
- **Architecture**: Layered architecture with separated API/Service/Storage layers

## Key Development Commands

### Backend Development

```bash
# Start backend with auto-reload
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest

# Run specific test module
pytest tests/test_detection.py -v

# Run with coverage
pytest --cov=app tests/

# Test spaCy sentence splitting
python test_chinese_sentence_split.py

# Test cross-encoder service
python scripts/test_cross_encoder.py --compare
```

### Frontend Development

```bash
# Start frontend dev server
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Lint check
cd frontend && npm run lint

# Preview production build
cd frontend && npm run preview
```

### Initial Setup

```bash
# Backend setup
pip install -r requirements.txt
python install_spacy_models.py
cp .env.example .env

# Frontend setup
cd frontend && npm install
```

## High-Level Architecture

### Detection Pipeline Architecture

The system uses a sophisticated multi-stage detection pipeline:

1. **Text Processing Stage**
   - Document parsing (PDF, DOCX, DOC, TXT, MD)
   - Intelligent sentence segmentation using spaCy
   - Chinese: min 8 tokens, title/honorific detection
   - English: min 6 tokens, abbreviation handling

2. **Embedding Generation Stage**
   - OpenAI-compatible API (default: Qwen3-Embedding-8B)
   - 4096-dimensional vectors
   - Batch processing (max 20 texts per batch)

3. **Similarity Detection Stage**
   - **AggressivePipelineService**: Primary detection engine
   - Multiple detection modes: PURE_SEMANTIC, AGGRESSIVE, FAST, STRICT
   - Optional MinHash for lexical similarity
   - Cross-encoder reranking (OpenAI or Jina providers)

4. **Match Filtering Stage**
   - **BidirectionalMatchFilter**: Ensures mutual best matches
   - Match strategies: `bidirectional_stable` (default), `hungarian`, `greedy`
   - Text alignment for precise span extraction

### Frontend Architecture

```
App.tsx
├── Routes (react-router-dom not used, custom state management)
│   ├── ProjectsPage → Project management
│   ├── TasksPage → Document library
│   ├── TaskDetailPage → Compare job management
│   ├── ProjectDetailPanel → Project documents/jobs
│   ├── ProjectJobPanel → Job pairs
│   └── PlanComparePage → Interactive comparison view
└── Components
    ├── layout/Page → PageShell, PageHeader, PageContent
    ├── ui/ → Radix UI based components
    └── progress/ → Real-time progress tracking
```

### Service Layer Organization

```
backend/services/
├── detection_orchestrator.py → Main orchestration logic
├── aggressive_similarity_pipeline.py → Core detection engine
├── text_processor.py → spaCy-based text processing
├── embedding_service.py → OpenAI API integration
├── cross_encoder_service.py → Reranking service
├── openai_reranker.py → OpenAI-compatible reranking
├── vector_storage.py → Milvus integration
├── bidirectional_match_filter.py → Match filtering
├── text_alignment.py → Precise span extraction
└── progress_tracker.py → WebSocket progress updates
```

## Critical Configuration

### Environment Variables (.env)

```bash
# OpenAI-compatible embedding API
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.vect.one/v1
EMBEDDING_MODEL=Qwen3-Embedding-8B
EMBEDDING_DIMENSIONS=4096

# Milvus storage
MILVUS_MODE=local  # or "server" for production
MILVUS_DB_FILE=milvus_demo.db

# Cross-encoder reranking
RERANKER_PROVIDER=openai  # or "jina"
RERANKER_MODEL=Qwen3-Reranker-0.6B
JINA_API_KEY=your-jina-key  # if using Jina

# Detection thresholds
PARAGRAPH_SIMILARITY_THRESHOLD=0.75
SENTENCE_SIMILARITY_THRESHOLD=0.80
```

### Detection Modes

Four pre-configured modes in `backend/models/detection_modes.py`:

1. **PURE_SEMANTIC**: Embeddings only, fastest
2. **AGGRESSIVE**: Full pipeline with MinHash + Cross-encoder
3. **FAST**: Semantic with light filtering
4. **STRICT**: All filters with high thresholds

## API Endpoints

### Core Endpoints

- `POST /api/v1/documents/upload` → Upload documents to project
- `GET /api/v1/documents/{document_id}` → Get document details
- `POST /api/v1/compare/jobs` → Create comparison job
- `POST /api/v1/compare/pairs` → Create document pairs
- `GET /api/v1/compare/pairs/{pair_id}/report` → Get comparison results
- `WS /api/v1/progress/ws/{task_id}` → Real-time progress updates

### Comparison Modes

- `chunk_type`: "sentence" or "paragraph"
- `mode`: Detection mode (see above)
- `match_strategy`: "bidirectional_stable", "hungarian", or "greedy"

## Development Patterns

### Frontend State Management

The app uses React hooks for state management without Redux/Context:
- `useDocuments`, `useProjects`, `useCompareJobs` → Data fetching hooks
- `useProgressTracking` → WebSocket progress tracking
- Component-level state for UI interactions

### Backend Async Patterns

All services use async/await for I/O operations:
- Database queries wrapped in `asyncio`
- Batch processing for embeddings
- Concurrent text processing with limits

### Error Handling

- Frontend: Axios interceptors + error boundaries
- Backend: FastAPI exception handlers + structured logging
- Progress tracking: Graceful WebSocket disconnection handling

## Testing Approach

### Backend Testing
- Unit tests for individual services
- Integration tests for API endpoints
- Performance tests for detection pipeline

### Frontend Testing
- Component testing with React Testing Library
- E2E testing considerations (not implemented yet)

### Key Test Commands
```bash
# Test detection pipeline
python scripts/test_full_workflow.py

# Test match strategies
python scripts/test_match_strategies.py

# Test bidirectional filtering
python scripts/test_bidirectional_strategy.py
```