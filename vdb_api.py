import os
import logging
from typing import List
import numpy as np

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import DataType, Property, Configure
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from enum import Enum

from InstructorEmbedding import INSTRUCTOR

# ------------------------------------------------------------------------------
# Load Environment & Setup Logger
# ------------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Enum for Embedding Types
# ------------------------------------------------------------------------------
class EmbeddingType(str, Enum):
    INSTRUCTOR = "instructor"
    BGE = "bge"
    OPENAI = "openai"

# ------------------------------------------------------------------------------
# Vector DB Manager
# ------------------------------------------------------------------------------
class VectorDBManager:
    def __init__(self, model):
        self.model = model
        self.vector_size = 768

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        assert self.client.is_ready(), "Weaviate client not ready"
        self._setup_collections()

    def _setup_collections(self):
        for embedding_type in EmbeddingType:
            collection_name = f"EmbeddingType_{embedding_type.value}"

            if self.client.collections.exists(collection_name):
                logger.info(f"✅ Collection {collection_name} already exists")
                continue

            self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="text_content", data_type=DataType.TEXT),
                    Property(name="embedding_instruction", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    vector_size=self.vector_size
                )
            )
            logger.info(f"🆕 Created collection {collection_name}")

    def _get_embedding(self, text: str, instruction: str) -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(self.vector_size)

        try:
            embedding = self.model.encode([[instruction, text]])
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.vector_size)

    def _document_exists(self, doc_id: str, doc_type: str, embedding_type: str) -> bool:
        collection_name = f"EmbeddingType_{embedding_type}"
        if not self.client.collections.exists(collection_name):
            return False

        collection = self.client.collections.get(collection_name)
        filters = (
            Filter.by_property("doc_id").equal(doc_id) &
            Filter.by_property("doc_type").equal(doc_type)
        )

        results = collection.query.fetch_objects(filters=filters)
        return len(results.objects) > 0

    def _add_to_weaviate(self, embedding_type: str, doc_id: str, doc_type: str, text: str, instruction: str):
        if self._document_exists(doc_id, doc_type, embedding_type):
            logger.info(f"🛑 Document {doc_id}-{doc_type} already exists in {embedding_type}")
            return

        embedding = self._get_embedding(text, instruction)
        collection_name = f"EmbeddingType_{embedding_type}"
        collection = self.client.collections.get(collection_name)
        collection.data.insert(
            properties={
                "doc_id": doc_id,
                "doc_type": doc_type,
                "text_content": text,
                "embedding_instruction": instruction,
            },
            vector=embedding,
        )
        logger.info(f"✅ Inserted document {doc_id}-{doc_type} into {embedding_type}")

    def get_embeddings_by_doc_id(self, embedding_type: str, doc_id: str, doc_type: str) -> List[List[float]]:
        collection_name = f"EmbeddingType_{embedding_type}"
        if not self.client.collections.exists(collection_name):
            return []

        collection = self.client.collections.get(collection_name)
        filters = (
            Filter.by_property("doc_id").equal(doc_id) &
            Filter.by_property("doc_type").equal(doc_type)
        )

        results = collection.query.fetch_objects(filters=filters)
        return [obj.vector for obj in results.objects if obj.vector]

    def get_resume_embeddings(self, resume_id: str, embedding_type: str) -> List[List[float]]:
        return self.get_embeddings_by_doc_id(embedding_type, resume_id, "resume")

    def get_job_embeddings(self, job_id: str, embedding_type: str) -> List[List[float]]:
        return self.get_embeddings_by_doc_id(embedding_type, job_id, "job")

    def close(self):
        self.client.close()
        logger.info("🔒 Weaviate client connection closed.")

# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        model = INSTRUCTOR("hkunlp/instructor-large")  # or instructor-base
        manager = VectorDBManager(model=model)

        # # Add resume sample
        # manager._add_to_weaviate(
        #     embedding_type="instructor",
        #     doc_id="resume_001",
        #     doc_type="resume",
        #     text="Experienced backend engineer with strong Python skills.",
        #     instruction="Represent the candidate's experience."
        # )

        # Retrieve
        embs = manager.get_resume_embeddings("resume_001", "resume")
        print("🔎 Retrieved Embeddings:", embs)

    finally:
        manager.close()
