# Handover
Handover is a secure Flask-based app that turns local documents into a private Q&amp;A system. Built with Python, Flask, and modern NLP embeddings, it enables semantic search and natural-language answers through a lightweight, self-hosted web interface.

Handover is a secure, Flask-powered document intelligence platform that transforms your local files into a private Q&A system. Built with Python, Flask, SQLite, and OpenAI API or Sentence-Transformers embeddings, it merges cutting-edge NLP, semantic search, and OCR technologies into a single lightweight web app. Just drop your files in the data/ folder, run the app, and start asking questions naturally.

# Features

Flask web application with authentication (login/logout, session management)

Natural language Q&A over locally stored documents

Multi-format ingestion: .pdf, .docx, .txt, .md, .xlsx, .xls, .png/.jpg (via OCR)

Local embeddings (Sentence-Transformers) or OpenAI integration

SQLite database for logging and user data

Clean UI powered by Flask templates and JSON APIs

Fully local — no cloud uploads unless OpenAI API is enabled

# How It Works

Place your files in the data/ directory.

Run handover.py ingest ./data to parse and index all supported documents.

Launch the Flask app (python app.py).

Log in and ask natural-language questions through the web interface.

The application searches embeddings locally (or via OpenAI if enabled), retrieves the most relevant text segments, and generates context-aware answers with citations.

# Author
Anna Zacharaki

