import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# The document set (or corpus) for reference. Each entry is a document.
DOCUMENTS = [
    "Transformers use self-attention to process text in parallel.",
    "TF-IDF gives higher weight to rare but important words.",
    "Word embeddings represent words as dense semantic vectors.",
    "RAG systems retrieve external documents before generating answers.",
    "CNNs are commonly used for image classification and object detection.",
    "Medical imaging AI can assist with segmentation and diagnosis.",
    "Large language models predict the next token during pretraining.",
    "Vector databases store embeddings for semantic search.",
]

def rank_results(scores, docs, top_k=3):
    """Similarity scores are sorted in descending order (highest to lowest) and top three are returned by default."""
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], scores[i]) for i in ranked_indices]


def vector_search(v, query, docs):
    """Fit a vectorizer on given corpus, embed user query using same vectorizer, and return the cosine similarity score 
    of query against each document in corpus"""
    x = v.fit_transform(docs)
    q = v.transform([query])
    scores = cosine_similarity(q, x).flatten()
    return scores

def keyword_search(query, docs, top_k=3):
    """BoW keyword search using count vectorizer"""
    vectorizer = CountVectorizer()
    scores = vector_search(vectorizer, query, docs)
    return rank_results(scores, docs, top_k)

def tfidf_search(query, docs, top_k=3):
    """TF-IDF keyword search using tfidf vectorizer"""
    vectorizer = TfidfVectorizer()
    scores = vector_search(vectorizer, query, docs)
    return rank_results(scores, docs, top_k)

def embedded_search(model, query, docs, top_k=3):
    """Use Word2Vec embedding model to perform search"""
    doc_embeddings = model.encode(docs)
    query_embeddings = model.encode([query])
    scores = cosine_similarity(query_embeddings, doc_embeddings).flatten()
    return rank_results(scores, docs, top_k)

def print_results(title, results):
    """Print the results on the standard output"""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"{i}. Score: {score:.4f}")
        print(f"{doc}")

def main():
    # Load a transformer-based MiniLM embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    while True:
        query = input("Search query: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            """User can enter 'exit' or 'quit' to exit from the search engine"""
            break
        
        # BoW search
        print_results(
            title="Bag-of-Words Search",
            results=keyword_search(query, DOCUMENTS)
        )

        # TF-IDF search
        print_results(
            title="TF-IDF Search",
            results=tfidf_search(query, DOCUMENTS)
        )

        # Embedding search
        print_results(
            title="Embeddings Search",
            results=embedded_search(model, query, DOCUMENTS)
        )

if __name__ == "__main__":
    main()
