# Customer Support Chatbot

A small Python portfolio project that retrieves answers from a curated FAQ dataset using **TF-IDF** and **cosine similarity**, with a Streamlit chat interface.

> **Important:** This is an independent demonstration project. The sample FAQ content is generic and is not official FOREO documentation, product advice, or an endorsed company project.

## Features

- FAQ knowledge base stored in JSON
- Text vectorization with TF-IDF
- Similarity ranking with cosine similarity
- Confidence threshold and fallback response
- Streamlit chat interface
- Basic automated tests

## Requirements

- Python 3.10–3.13 recommended
- pip

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the web app

From the project root:

```bash
streamlit run app.py
```

## Run in the terminal

```bash
python chatbot.py
```

Type `quit` to exit.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## How it works

1. FAQ questions and keywords are combined into searchable text.
2. `TfidfVectorizer` converts the FAQ text into numerical vectors.
3. The user's question is transformed using the same vectorizer.
4. Cosine similarity ranks the FAQs.
5. If the highest score is below the configured threshold, the bot returns a safe fallback instead of guessing.

This is a retrieval-based NLP chatbot, not a generative AI model. It does not train a machine-learning model on customer data and does not call an external AI API.

## Project structure

```text
ai-customer-support-chatbot/
├── app.py
├── chatbot.py
├── requirements.txt
├── README.md
├── data/
│   └── knowledge_base.json
└── tests/
    └── test_chatbot.py
```

## Possible next improvements

- Add a larger, reviewed FAQ dataset
- Evaluate precision and recall on a labelled test set
- Add multilingual support
- Add an optional LLM API with clear privacy and cost controls
- Add logging and feedback collection without storing sensitive personal data
