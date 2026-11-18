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