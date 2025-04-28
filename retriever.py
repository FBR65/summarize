import logging
from typing import List, Dict, Optional

from fastembed import SparseTextEmbedding
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from __init__ import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    BASE_URL,
    API_KEY,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    SPARSE_MODEL,
    SUMMARIZATION_MODEL,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoredPoint(BaseModel):
    id: str = Field(description="Unique identifier for the point")
    version: int = Field(description="Version of the point")
    score: float = Field(description="Score of the point")
    payload: Dict = Field(description="Payload data")
    vector: Optional[List[float]] = Field(
        None, description="Vector representation of the point"
    )
    shard_key: Optional[str] = Field(None, description="Shard key for the point")
    order_value: Optional[float] = Field(None, description="Order value for the point")


class Hybrid_search:
    """
    A class for performing hybrid search using dense and sparse embeddings,
    reranking, and summarizing retrieved texts.
    """

    def __init__(self, collection_name_prefix) -> None:
        """
        Initialize the Hybrid_search object with models, clients,
        and the collection name prefix.
        """
        self.collection_name_prefix = collection_name_prefix
        # Dieser Client wird für Embeddings, Reranking und Summarization verwendet
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,  # required but ignored by some local servers
        )
        self.embedding_model = EMBEDDING_MODEL
        self.reranker_model = RERANKER_MODEL
        self.sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        self.summarization_model = SUMMARIZATION_MODEL

        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )

    def rerank_results(
        self, query: str, results: List[models.ScoredPoint]
    ) -> List[models.ScoredPoint]:
        """
        Reranks the given search results using a reranker model.
        (Unverändert)
        """
        if not results:
            return []

        # Sicherstellen, dass 'text' im Payload vorhanden ist
        pairs = []
        valid_results = []
        for result in results:
            if result.payload and "text" in result.payload:
                pairs.append((query, result.payload["text"]))
                valid_results.append(result)
            else:
                logger.warning(
                    f"Skipping result {result.id} for reranking due to missing 'text' in payload."
                )

        if not pairs:
            return []

        # Prepare input for the reranker model
        reranker_input = [f"query: {pair[0]} document: {pair[1]}" for pair in pairs]

        try:
            # Call the reranker model
            response = self.client.embeddings.create(
                model=self.reranker_model, input=reranker_input
            )

            # Extract the rerank scores from the response
            rerank_scores = [data.embedding[0] for data in response.data]

            # Sort results by the rerank scores
            reranked_results = sorted(
                zip(valid_results, rerank_scores),
                key=lambda item: item[1],
                reverse=True,
            )
            return [result for result, score in reranked_results]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback: Rückgabe der gültigen, aber nicht neu gerankten Ergebnisse
            return valid_results

    def query_hybrid_search(self, query, metadata_filter=None, limit=20):
        """
        Performs hybrid search across all distance collections and returns raw, reranked per collection,
        and reranked combined results.
        (Unverändert bis auf Logging und Fehlerbehandlung)
        """
        try:
            response = self.client.embeddings.create(
                input=[query], model=self.embedding_model
            )
            dense_query = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting dense embeddings: {e}")
            return {  # Leere Ergebnisse zurückgeben bei Fehler
                "raw_results_per_collection": {},
                "reranked_results_per_collection": {},
                "deduplicated_combined_results": [],
            }

        try:
            sparse_query = list(self.sparse_embedding_model.embed([query]))[0]
        except Exception as e:
            logger.error(f"Error getting sparse embeddings: {e}")
            return {  # Leere Ergebnisse zurückgeben bei Fehler
                "raw_results_per_collection": {},
                "reranked_results_per_collection": {},
                "deduplicated_combined_results": [],
            }

        collection_distances = ["COSINE"]  # Beispiel, ggf. anpassen
        all_raw_results = []
        raw_results_per_collection = {}
        reranked_results_per_collection = {}

        for distance_type in collection_distances:
            collection_name = f"{self.collection_name_prefix}_{distance_type}"
            logger.info(f"Performing hybrid search on collection: {collection_name}")
            try:
                # Sicherstellen, dass die Vektoren die korrekte Struktur haben
                sparse_vector_input = models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                )
                dense_vector_input = dense_query  # Ist bereits eine Liste von Floats

                raw_results = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=sparse_vector_input,
                            using="sparse",
                            limit=limit,
                        ),
                        models.Prefetch(
                            query=dense_vector_input,
                            using="dense",
                            limit=limit,
                        ),
                    ],
                    query_filter=metadata_filter,
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF  # Reciprocal Rerank Fusion
                    ),
                    limit=limit,
                    # Wichtig: Payload und Vektoren für Reranking/Summarization abrufen
                    with_payload=True,
                    # with_vectors=True # Nur nötig, wenn Vektoren explizit gebraucht werden
                )
                # Konvertiere qdrant_client.http.models.ScoredPoint in unsere Pydantic-Klasse
                processed_points = [
                    ScoredPoint(**point.model_dump()) for point in raw_results.points
                ]

                raw_results_per_collection[collection_name] = processed_points
                all_raw_results.extend(processed_points)

                # Rerank results for the current collection
                reranked_collection_results = self.rerank_results(
                    query, processed_points
                )
                reranked_results_per_collection[collection_name] = (
                    reranked_collection_results
                )

            except Exception as e:
                logger.error(
                    f"Error during hybrid search on collection {collection_name}: {e}"
                )
                raw_results_per_collection[collection_name] = []
                reranked_results_per_collection[collection_name] = []

        # Deduplizieren der kombinierten rohen Ergebnisse basierend auf der ID
        deduplicated_raw_results = []
        seen_ids = set()
        for result in all_raw_results:
            if result.id not in seen_ids:
                deduplicated_raw_results.append(result)
                seen_ids.add(result.id)

        # Rerank die deduplizierten Ergebnisse
        reranked_deduplicated_results = self.rerank_results(
            query, deduplicated_raw_results
        )

        return {
            "raw_results_per_collection": raw_results_per_collection,
            "reranked_results_per_collection": reranked_results_per_collection,
            "deduplicated_combined_results": reranked_deduplicated_results,
        }

    def summarize_texts(
        self, texts: List[str], max_tokens: int = 200, query: Optional[str] = None
    ) -> str:
        """
        Summarizes a list of texts using an OpenAI compatible chat model,
        optionally considering the original query.

        Args:
            texts (List[str]): A list of text strings to summarize.
            max_tokens (int): The maximum number of tokens for the summary.
            query (Optional[str]): The original search query to focus the summary.

        Returns:
            str: The generated summary or an error message.
        """
        if not texts:
            return "Kein Text zur Zusammenfassung vorhanden."

        # Texte zu einem einzigen Dokument zusammenfügen
        # Trennzeichen helfen dem Modell, die einzelnen Dokumente zu unterscheiden
        full_text = "\n\n---\n\n".join(texts)

        # System-Prompt definieren
        system_prompt = (
            "Du bist ein hilfreicher Assistent, der Texte prägnant zusammenfasst."
        )

        # User-Prompt erstellen
        user_prompt_parts = ["Bitte fasse die folgenden Textabschnitte zusammen."]
        if query:
            user_prompt_parts.append(
                f"Konzentriere dich dabei auf Aspekte, die relevant für die Frage '{query}' sind."
            )
        user_prompt_parts.append("\n\nTextabschnitte:\n")
        user_prompt_parts.append(full_text)
        user_prompt_parts.append("\n\nZusammenfassung:")
        user_prompt = "\n".join(user_prompt_parts)

        try:
            # Verwende die Chat-Completion-API
            response = self.client.chat.completions.create(
                model=self.summarization_model,  # Modellnamen ggf. anpassen
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Niedrigere Temperatur für faktenbasierte Zusammenfassung
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Error during text summarization: {e}")
            # Versuchen, eine spezifischere Fehlermeldung zu geben
            error_message = str(e)
            if "context length" in error_message.lower():
                return f"Fehler bei der Zusammenfassung: Der Text ist zu lang für das Modell ({self.summarization_model}). Reduzieren Sie die Anzahl der Dokumente (limit) oder kürzen Sie die Texte."
            return f"Fehler bei der Zusammenfassung: {e}"


if __name__ == "__main__":
    # Sicherstellen, dass der Prefix korrekt ist für die gewünschte Collection
    collection_name_prefix = "sum_collection"
    hybrid_search = Hybrid_search(collection_name_prefix)

    # Beispiel-Suchanfrage
    search_query = "Was soll im Datenschutz gemacht werden?"
    # Limit für die Anzahl der abgerufenen und zusammenzufassenden Dokumente
    limit = 5  # Reduzieren Sie dies, wenn Kontextlängenprobleme auftreten

    # 1. Daten aus Qdrant abrufen (Hybrid Search + Reranking)
    print(f"Führe hybride Suche für '{search_query}' durch (Limit: {limit})...")
    hybrid_results = hybrid_search.query_hybrid_search(search_query, limit=limit)

    # Extrahieren der Texte aus den besten (deduplizierten, rerankten) Ergebnissen
    retrieved_texts = []
    print("\nTop deduplizierte & rerankte Ergebnisse:")
    if hybrid_results["deduplicated_combined_results"]:
        for i, result in enumerate(hybrid_results["deduplicated_combined_results"]):
            # Zeige nur die Top-Ergebnisse an (optional)
            # if i >= limit: # Zeige nur so viele wie das Limit war
            #     break
            print(f"  Rank {i + 1}: ID={result.id}, Score={result.score:.4f}", end="")
            if result.payload and "text" in result.payload:
                # Zeige einen Snippet des Textes
                text_snippet = result.payload["text"][:100].replace("\n", " ") + "..."
                print(f", Text Snippet: '{text_snippet}'")
                retrieved_texts.append(result.payload["text"])
            else:
                print(", Payload enthält keinen 'text'.")
                logger.warning(f"Payload für ID {result.id} enthält kein 'text'-Feld.")
    else:
        print("  Keine Ergebnisse gefunden.")

    # 2. Die abgerufenen Texte zusammenfassen
    if retrieved_texts:
        print(f"\nFasse {len(retrieved_texts)} abgerufene Dokumente zusammen...")
        # Übergabe der ursprünglichen Query für eine fokussiertere Zusammenfassung
        summary = hybrid_search.summarize_texts(
            retrieved_texts, max_tokens=250, query=search_query
        )
        print("\n--- Generierte Zusammenfassung ---")
        print(summary)
        print("------------------------------")
    else:
        print(
            "\nKeine Texte in den Ergebnissen gefunden, die zusammengefasst werden könnten."
        )
