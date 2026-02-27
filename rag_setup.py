import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import requests
import json

# ====== 1. Connect to Elastic Cloud ======

ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

es = Elasticsearch(
    ELASTIC_URL,
    api_key=ELASTIC_API_KEY
)

print("Connected to Elasticsearch!")

# ====== 2. Load Embedding Model ======
model = SentenceTransformer("all-MiniLM-L6-v2")

# ====== 3. Sample Documents ======
documents = [
    "Elasticsearch is a distributed search and analytics engine built for scalability.",
    "Vector search enables semantic similarity using dense vector embeddings.",
    "Retrieval-Augmented Generation reduces hallucinations in LLMs.",
    "Elasticsearch supports hybrid search combining BM25 and vector similarity.",
    "Cosine similarity measures the angle between embedding vectors."
]

# ====== 4. Index Documents with Embeddings ======
for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()

    es.index(
        index="chat_index",
        id=i,
        document={
            "content": doc,
            "embedding": embedding
        }
    )

print("Documents indexed successfully!")

# ====== 5. Perform Vector Search ======

query = "How does Elasticsearch improve search?"
query_embedding = model.encode(query).tolist()

response = es.search(
    index="chat_index",
    query={
        "bool": {
            "should": [
                {
                    "match": {
                        "content": query
                    }
                }
            ]
        }
    },
    knn={
        "field": "embedding",
        "query_vector": query_embedding,
        "k": 2,
        "num_candidates": 10
    }
)

print("\nSearch Results:\n")

for hit in response["hits"]["hits"]:
    print(f"Score: {hit['_score']}")
    print(f"Content: {hit['_source']['content']}")
    print("-" * 40)

# ====== 6. Build Simple RAG Response ======

retrieved_docs = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
context = "\n".join(retrieved_docs)

final_prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

print("\nConstructed Prompt for LLM:\n")
print(final_prompt)

# ====== 7. Call Gemini API (API key via environment variable) ======

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {"text": final_prompt}
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
result = response.json()

print("\nFinal Answer:\n")
print(result["candidates"][0]["content"]["parts"][0]["text"])
