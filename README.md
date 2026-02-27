# Hybrid RAG Assistant with Elasticsearch (Elastic Blogathon 2026)

This project demonstrates how to build a **Hybrid Retrieval-Augmented Generation (RAG) system** using:

- Elasticsearch as a vector database  
- Elastic Cloud deployment  
- Dense vector indexing (384-dim)  
- Hybrid search (BM25 + kNN)  
- HNSW ANN indexing  
- Gemini 2.5 Flash for generation  
- Python implementation  

The system retrieves relevant documents using hybrid search and generates grounded answers using an LLM.

---

## 🚀 Architecture Overview

The system follows this pipeline:

1. User submits a query  
2. Query is converted into a 384-dimensional embedding using MiniLM  
3. Elasticsearch performs:  
   - BM25 keyword search  
   - kNN vector search (HNSW + cosine similarity)  
4. Top results are combined into context  
5. Gemini generates a grounded response  

---

## 🧠 Why Hybrid Search?

- Keyword search provides lexical precision  
- Vector search provides semantic understanding  
- Hybrid search balances both  

This improves retrieval quality and reduces hallucination in LLM responses.

---

## 🏗 Tech Stack

- Python  
- Elasticsearch (Elastic Cloud)  
- Sentence Transformers (`all-MiniLM-L6-v2`)  
- Gemini 2.5 Flash API  
- HNSW vector indexing  

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install elasticsearch sentence-transformers requests
```

---

## 🔐 Environment Variables

⚠️ **Do NOT hardcode API keys.**

Set the following environment variables:

### Windows (PowerShell)

```powershell
setx ELASTIC_URL "your_elastic_cloud_url"
setx ELASTIC_API_KEY "your_elastic_api_key"
setx GEMINI_API_KEY "your_gemini_api_key"
```

### macOS / Linux

```bash
export ELASTIC_URL="your_elastic_cloud_url"
export ELASTIC_API_KEY="your_elastic_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

---

## 🗂 Elasticsearch Index Setup

Create the vector index:

```json
PUT chat_index
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

This enables HNSW-based vector search.

---

## ▶️ Running the Project

```bash
python rag_Setup.py
```

The script will:

- Connect to Elasticsearch  
- Generate embeddings  
- Index documents  
- Perform hybrid search  
- Construct RAG prompt  
- Generate final LLM response  

---

## 📊 Example Output

```
Search Results:

Score: 2.20
Content: Elasticsearch supports hybrid search combining BM25 and vector similarity.

Score: 2.12
Content: Elasticsearch is a distributed search and analytics engine built for scalability.
```

Final Generated Answer:

```
Elasticsearch improves search by combining BM25 keyword matching with vector similarity,
allowing both lexical precision and semantic relevance.
```

---

## 🔎 How Hybrid Search Works

Elasticsearch query structure:

```json
GET chat_index/_search
{
  "query": {
    "bool": {
      "must": {
        "match": {
          "content": "Elasticsearch search"
        }
      }
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [384-dimensional vector],
    "k": 3,
    "num_candidates": 10
  }
}
```

This merges lexical scoring with semantic similarity.

---

## 🌍 Real-World Use Cases

- Enterprise knowledge assistants  
- AI customer support bots  
- E-commerce semantic search  
- Observability log analysis  
- AI copilots grounded in proprietary data  

---

## 📈 Production Considerations

- HNSW enables scalable ANN search  
- Elastic Cloud supports horizontal scaling  
- Hybrid scoring improves retrieval precision  
- RAG reduces hallucination  

---

## 🛠 Future Improvements

- Add cross-encoder reranking  
- Benchmark latency at scale  
- Add vector quantization  
- Deploy as REST API  
- Add web interface  

---

## 🏆 Elastic Blogathon 2026

This project was built as part of:

**Elastic Blogathon 2026 – Vectorized Thinking**

It demonstrates how Elasticsearch can function as a scalable vector database powering AI-driven applications.

---

## 📄 License

MIT License
