from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tika import parser
from datetime import datetime as dt
import logging
import secrets
import uuid
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from openai import OpenAI
from __init__ import (
    TIKA_SERVER_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    BASE_URL,
    API_KEY,
    EMBEDDING_MODEL,
    SPARSE_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)


"""
    Be sure to use the spaCy model and the sentence transformer model which is best
    for your language.

    If you are using uv (as I'm) you are able to install the models right from
    spaCy GitHub:

    
    uv pip install de_core_news_lg@https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl   
"""


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """
    Is giving back Document(inhalt=content, metadaten=metadata)
    Needed to get a structured output for the parsed Document
    """

    def __init__(self, inhalt: str, metadaten: dict = None):
        self.inhalt = inhalt
        self.metadaten = metadaten if metadaten is not None else {}


def semantic_chunking(text, threshold_percentile=25):
    # Schritt 1: Sätze aufteilen
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Schritt 2: Embeddings generieren
    embeddings = model.encode(sentences)

    # Schritt 3: Ähnlichkeiten berechnen
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)

    # Schritt 4: Breakpoints ermitteln
    threshold = np.percentile(similarities, threshold_percentile)

    # Schritt 5: Chunks erstellen
    chunks, current_chunk = [], []
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        if i < len(similarities) and similarities[i] < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class DataIntake:
    def __init__(self, collection_name, file_path) -> None:
        """
        Initialize the data_intake object with dense and sparse embedding models and a Qdrant client.
        """
        self.collection_name = collection_name
        self.sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,  # required but ignored
        )
        self.embedding_model = EMBEDDING_MODEL
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )
        self.file_path = file_path
        global model
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
        global nlp
        nlp = spacy.load("de_core_news_lg")

    def organize_intake(self):
        logging.info(f"Intake process started at: {dt.now()}")
        text = self.stream_document(self.file_path)
        logging.info(f"Stream Document ended at: {dt.now()}")
        logging.info(f"Create Collection started: {dt.now()}")
        create_collection = self.client_collection()  # Call the method
        logging.info(f"Create Collection ended at: {dt.now()}")
        if create_collection:
            logging.info(f"Chunk Splitting started at: {dt.now()}")
            chunk_files = self.split_into_chunks(
                text.inhalt
            )  # Split into chunks and save to files
            logging.info(f"Chunk Splitting ended at: {dt.now()}")
            logging.info(f"DB Intake started at: {dt.now()}")
            self.fill_database(chunk_files)  # Pass list of chunk file paths
            logging.info(f"DB Intake ended at: {dt.now()}")
        logging.info(f"Intake process ended at: {dt.now()}")
        return f"Finished at: {dt.now()}"

    def generate_point_id(self):
        """
        Make sure the point id is unique.
        Otherwise, an existing point will be overwritten with a further intake.
        """
        # generate UUID
        uuid_value = uuid.uuid4().hex

        # replace digits through random values
        modified_uuid = "".join(
            (
                hex((int(c, 16) ^ secrets.randbits(4) & 15 >> int(c) // 4))[2:]
                if c in "018"
                else c
            )
            for c in uuid_value
        )

        logging.info(f"Created point id '{modified_uuid}'.")

        return str(modified_uuid)

    def split_into_chunks(self, text, output_dir="temp_chunks"):
        """
        Split the document into chunks and save each chunk to a JSON file.
        Returns a list of file paths to the saved chunks.
        """
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error creating directory '{output_dir}': {e}")

        chunk_files = []
        chunks = semantic_chunking(text)
        chunk_index = 0
        for chunk in chunks:
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_index}.json")
            with open(chunk_filename, "w", encoding="utf-8") as f:
                json.dump({"text": chunk}, f)
            chunk_files.append(chunk_filename)
            chunk_index += 1
        logging.info(f"Saved {len(chunk_files)} chunks to '{output_dir}'.")
        return chunk_files

    def stream_document(self, path):
        """
        Sending the Document to Apache Tika Server.
        Document will be parsed and the Content and Metadata will be given back.
        """
        logging.info("Stream Document started.")
        parsed = parser.from_file(path, serverEndpoint=TIKA_SERVER_URL)

        if "resourceName" in parsed["metadata"]:
            if isinstance(parsed["metadata"]["resourceName"], list):
                decoded_text = parsed["metadata"]["resourceName"][0].strip("b'")
            else:
                decoded_text = parsed["metadata"]["resourceName"].strip("b'")

            parsed["metadata"]["file_name"] = decoded_text
            del parsed["metadata"]["resourceName"]
        content = parsed["content"]
        metadata = parsed["metadata"]
        document = Document(inhalt=content, metadaten=metadata)

        return document

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        The collection will be created with all possible dense vector distances
        These will be used in the corrosponding agent system
        """
        collection_distances = ["COSINE"]

        for distances in collection_distances:
            collection = self.collection_name + "_" + str(distances)

            match distances:
                case "COSINE":
                    distance = models.Distance.COSINE
                case "EUCLID":
                    distance = models.Distance.EUCLID
                case "DOT":
                    distance = models.Distance.DOT
                case "MANHATTAN":
                    distance = models.Distance.MANHATTAN

            if not self.qdrant_client.collection_exists(
                collection_name=f"{collection}"
            ):
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=768,
                            distance=distance,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=False,
                            ),
                        )
                    },
                )
                logging.info(
                    f"Created collection '{collection}' in Qdrant vector database."
                )

                self.qdrant_client.create_payload_index(
                    collection_name=f"{collection}",
                    field_name="text",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=15,
                        lowercase=True,
                    ),
                )
                logging.info(f"Created payload index for collection '{collection}'.")
        return "created"

    def fill_database(self, chunk_file_paths):
        collection_distances = ["COSINE"]
        for file_path in chunk_file_paths:
            points = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    chunk = chunk_data["text"]

                response = self.client.embeddings.create(
                    input=[chunk], model=self.embedding_model
                )
                dense_embedding = response.data[0].embedding

                sparse_embedding_result = self.sparse_embedding_model.embed([chunk])

                sparse_embedding = None
                for embedding in sparse_embedding_result:
                    sparse_embedding = {
                        "indices": embedding.indices.tolist(),
                        "values": embedding.values.tolist(),
                    }
                    break

                point_id = self.generate_point_id()
                payload = {"text": chunk, "chunk_id": point_id}
                if dense_embedding is not None and sparse_embedding is not None:
                    points.append(
                        models.PointStruct(
                            id=point_id,  # the PointStruct needs an ID
                            vector={
                                "dense": dense_embedding,
                                "sparse": sparse_embedding,
                            },
                            payload=payload,
                        )
                    )
                elif dense_embedding is None:
                    logging.warning(
                        f"Dense embedding was None für Chunk: '{chunk[:50]}...'"
                    )
                elif sparse_embedding is None:
                    logging.warning(
                        f"Sparse embedding was None für Chunk: '{chunk[:50]}...'"
                    )
                else:
                    logging.warning(
                        f"Both Embeddings (dense und sparse) failed for : '{chunk[:50]}...'"
                    )

                if points:
                    for distances in collection_distances:
                        collection = self.collection_name + "_" + str(distances)
                        self.qdrant_client.upsert(
                            collection_name=collection, points=points, wait=True
                        )
                        logging.info(
                            f"Successfully uploaded chunk from '{file_path}' to '{collection}'."
                        )
                else:
                    logging.info(f"No data to upload for chunk file '{file_path}'.")

            except Exception as e:
                logging.error(f"Error during File '{file_path}': {e}")

        for file_path in chunk_file_paths:
            os.remove(file_path)
        if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
            os.rmdir("temp_chunks")


if __name__ == "__main__":
    file_path = "./data/41382-8.txt"
    collection_name = "sum_collection"
    intake = DataIntake(collection_name=collection_name, file_path=file_path)
    organized = intake.organize_intake()
    print(organized)
