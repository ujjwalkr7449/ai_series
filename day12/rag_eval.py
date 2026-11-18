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
