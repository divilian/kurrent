# Functions to compute embeddings for chunks of text.
import torch
from sentence_transformers import SentenceTransformer


def get_embed_model(embed_model: str) -> SentenceTransformer:
    return SentenceTransformer(embed_model,
        device="cuda" if torch.cuda.is_available() else "cpu")

def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """
    Convert a list of texts into normalized embeddings.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings
