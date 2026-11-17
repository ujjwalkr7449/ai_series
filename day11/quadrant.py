import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# Load variables from .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")

# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384


# Delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)


# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)
# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    
# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2") #384

print("Embedding model ready!")
texts = [document["text"] for document in documents]


embeddings = model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

points = []

for i in range(len(documents)):
    
    point = PointStruct(
        id=i + 1,
        vector=embeddings[i].tolist(),
        payload=documents[i]
    )

    points.append(point)
# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================

client.upsert( #upload+insert
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")