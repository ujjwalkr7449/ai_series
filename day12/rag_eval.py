# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
import json

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

from sentence_transformers import SentenceTransformer

from groq import Groq

# ============================================================
# PART 2 — LOAD ENVIRONMENT
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#============================================================
# PART 3 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud!")


# ============================================================
# PART 4 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "rag_eval"
EMBEDDING_SIZE = 384


if client.collection_exists(COLLECTION_NAME):

    print(
        f"Deleting existing collection: {COLLECTION_NAME}"
    )

    client.delete_collection(
        COLLECTION_NAME
    )


client.create_collection(

    collection_name=COLLECTION_NAME,

    vectors_config=VectorParams(

        size=EMBEDDING_SIZE,

        distance=Distance.COSINE
    )
)


print(
    f"Created collection: {COLLECTION_NAME}"
)
