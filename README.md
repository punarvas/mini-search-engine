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