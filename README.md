Mini Search Engine
----
The mini search engine let's user to find answer to their search query using various methods of text representation. This is a simple learning project.

Process:
- User enter the query
- For each algorithm, the program embeds the reference document and user query using the BoW, TF-IDF, and embedding model
- Based on the cosine similarity, top three results are ranked.
- Results are returned that contain score and the document that matched.
- All results are shown with scores


Requirements:
- Python 3.12 or above

How to run the code?
---
```
> cd mini-search-engine
> python3 -m venv .venv
(windows)> .venv\Scripts\activate
> pip install -r requirements.txt
> python app.py 
```

Output sample:
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<?, ?it/s]
Search query: how do llms generate text

============================================================
Bag-of-Words Search
============================================================
1. Score: 0.3333
Transformers use self-attention to process text in parallel.
2. Score: 0.0000
Vector databases store embeddings for semantic search.
3. Score: 0.0000
Medical imaging AI can assist with segmentation and diagnosis.

============================================================
TF-IDF Search
============================================================
1. Score: 0.3390
Transformers use self-attention to process text in parallel.
2. Score: 0.0000
Vector databases store embeddings for semantic search.
3. Score: 0.0000
Medical imaging AI can assist with segmentation and diagnosis.

============================================================
Embeddings Search
============================================================
1. Score: 0.3527
Transformers use self-attention to process text in parallel.
2. Score: 0.3057
RAG systems retrieve external documents before generating answers.
3. Score: 0.2380
Large language models predict the next token during pretraining.
```