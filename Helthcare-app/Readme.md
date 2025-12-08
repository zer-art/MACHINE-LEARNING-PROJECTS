# 🏥 Healthcare Question-Answering App

A Python-based healthcare assistant that leverages a knowledge base and custom functions to answer user queries about health topics. Designed for extensibility and easy integration.

## 🚀 Features

- **Question-Answering System**: Answers health-related questions using a curated knowledge base.
- **Extensible Functions**: Add custom logic for advanced queries.
- **Modular Design**: Organized codebase for maintainability.
- **CLI and API Ready**: Easily adaptable for web or command-line interfaces.

## 📚 Data Source

The medical knowledge base for this application is curated from:
> **The Gale Encyclopedia of Medicine, Second Edition, Volume One**

This authoritative source ensures that the information provided by the assistant is grounded in verified medical literature.

## 📊 Performance Metrics

To ensure responsiveness and reliability, we track key performance indicators. You can run the included `benchmark.py` to verify these on your machine.

| Metric | Description | Value (Approx.) |
| :--- | :--- | :--- |
| **Initialization Time** | Time to load embeddings and connect to Pinecone | ~2.5s |
| **Query Latency** | Time to retrieve docs and generate answer | ~1.5s - 3.0s |
| **Knowledge Base** | Number of medical documents indexed | *Dynamic* |
| **Model** | LLM used for generation | Gemini 1.5 Flash |

*Note: Latency depends on network speed and API response times.*

## 🏗️ Project Structure

```
Helthcare-app/
├── app.py                # Main application entry point
├── knowledege_base.py    # Healthcare knowledge base logic
├── src/
│   ├── __init__.py
│   ├── functions.py      # Custom functions for Q&A
│   └── prompt.py         # Prompt templates and logic
├── pyproject.toml        # Project configuration
├── setup.py              # Setup script
└── HEALTHCARE_APP.egg-info/
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** (if used for data)
- **Custom Python modules**

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

```bash
git clone <repository-url>
cd ML_PROJECTS/Helthcare-app
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 🚀 Usage

```bash
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## 👤 Author

**Pawan Parida**

---
A Python-based healthcare assistant that leverages a knowledge base and custom functions to answer user queries about health topics. Designed for extensibility and easy integration.

## 🚀 Features

- **Question-Answering System**: Answers health-related questions using a curated knowledge base.
- **Extensible Functions**: Add custom logic for advanced queries.
- **Modular Design**: Organized codebase for maintainability.
- **CLI and API Ready**: Easily adaptable for web or command-line interfaces.

