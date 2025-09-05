# FastAPI Vector Embedding Plagiarism Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green)
![Milvus](https://img.shields.io/badge/Milvus-2.4%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance plagiarism detection system using vector embeddings and semantic similarity search.

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [API Documentation](#api-documentation) • [Contributing](#contributing)

</div>

## 🚀 Features

- **Vector-Based Detection**: Uses advanced embedding models for semantic similarity detection
- **Two-Level Analysis**: Paragraph-level (fast) and sentence-level (detailed) detection modes
- **Dual Storage Modes**: Local file-based storage for development, Milvus server for production
- **OpenAI-Compatible API**: Supports any OpenAI-compatible embedding service
- **Real-time Processing**: Sub-100ms response time for most queries
- **RESTful API**: Clean, documented API with automatic Swagger/OpenAPI documentation
- **High Accuracy**: >85% detection accuracy with configurable similarity thresholds

## 📦 Tech Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **Milvus** - Vector database for similarity search
- **OpenAI API** - Text embeddings (compatible with any OpenAI-like API)
- **Pydantic** - Data validation using Python type annotations
- **Redis** - Caching layer (optional)
- **structlog** - Structured logging

## 🎯 Quick Start

### Prerequisites

- Python 3.8+
- pip or poetry
- OpenAI API key (or compatible embedding service)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plagiarism-detector.git
cd plagiarism-detector
```

2. **Initialize project structure**
```bash
python scripts/init_project.py
```

3. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your API credentials
```

5. **Run the application**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the API documentation**
```
http://localhost:8000/docs
```

## 🏗️ Architecture

The system follows a clean, layered architecture:

```
┌─────────────────────────────────────────┐
│          API Layer (FastAPI)            │
├─────────────────────────────────────────┤
│         Service Layer                   │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐│
│  │Embedding │ │Detection │ │Storage  ││
│  │Service   │ │Service   │ │Service  ││
│  └──────────┘ └──────────┘ └─────────┘│
├─────────────────────────────────────────┤
│       Repository Layer                  │
│  ┌──────────┐ ┌──────────┐            │
│  │  Milvus  │ │  Redis   │            │
│  └──────────┘ └──────────┘            │
└─────────────────────────────────────────┘
```

### Detection Flow

1. **Text Input** → Document submitted for plagiarism check
2. **Text Processing** → Split into paragraphs/sentences
3. **Embedding Generation** → Convert text to vectors using embedding model
4. **Similarity Search** → Find similar vectors in Milvus database
5. **Result Analysis** → Calculate similarity scores and return matches

## 📋 API Documentation

### Core Endpoints

#### Check Plagiarism
```http
POST /api/v1/detection/check
Content-Type: application/json

{
  "content": "Text to check for plagiarism...",
  "mode": "fast",  // Options: fast, detailed, comprehensive
  "threshold": 0.75
}
```

Response:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "total_matches": 5,
  "paragraph_matches": [...],
  "sentence_matches": [...],
  "processing_time": 0.089
}
```

#### Upload Document
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: document.pdf
```

#### Health Check
```http
GET /api/v1/health
```

### Detection Modes

- **Fast**: Paragraph-level detection only (< 50ms)
- **Detailed**: Paragraph + sentence analysis for high matches
- **Comprehensive**: Full analysis of all text segments

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for embeddings | Required |
| `OPENAI_BASE_URL` | Custom API endpoint | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Embedding model name | `text-embedding-3-large` |
| `MILVUS_MODE` | Storage mode (local/server) | `local` |
| `PARAGRAPH_SIMILARITY_THRESHOLD` | Paragraph match threshold | `0.75` |
| `SENTENCE_SIMILARITY_THRESHOLD` | Sentence match threshold | `0.80` |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_detection.py -v
```

## 📊 Performance Metrics

- **Response Time**: < 100ms (95th percentile)
- **Throughput**: > 100 requests/second
- **Accuracy**: > 85% detection rate
- **False Positive Rate**: < 5%
- **Concurrent Users**: > 100

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development Principles

This project follows Linus Torvalds' philosophy:
> "Talk is cheap. Show me the code."

- **简单性 (Simplicity)**: Keep it simple, avoid complexity
- **实用性 (Practicality)**: Make it work first, optimize later
- **清晰性 (Clarity)**: Code is documentation, naming is commentary

## 🐛 Known Issues

- Batch size limited to 20 for current embedding model
- Local mode (file-based) suitable for < 1M documents
- Sentence-level detection increases processing time by ~2x

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Development Guide](docs/development-roadmap.md) - Step-by-step development guide
- [Implementation Details](docs/comprehensive-implementation.md) - Full technical specification

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Milvus team for the powerful vector database
- OpenAI for embedding models and API design
- Contributors and testers

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<div align="center">
Made with ❤️ by the open source community
</div>