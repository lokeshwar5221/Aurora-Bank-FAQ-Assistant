"""Aurora Bank FAQ Assistant - Retrieval-Augmented Generation engine.

Modules (one per project feature):
    ingest.py       -> Policy Document Ingestion
    embedder.py     -> Embedding
    vectorstore.py  -> Indexing (vector database + similarity search)
    retriever.py    -> Retriever
    generator.py    -> Answer Generator (prompt templates + guardrails)
    pipeline.py     -> Ties ingestion + embedding + indexing into one KnowledgeBase
"""
