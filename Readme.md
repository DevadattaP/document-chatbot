# Document Chatbot using DistilBERT and FLAN-T5 with GloVe

## Table of Contents
- [Document Chatbot using DistilBERT and FLAN-T5 with GloVe](#document-chatbot-using-distilbert-and-flan-t5-with-glove)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Models Used](#models-used)
    - [DistilBERT-based Model](#distilbert-based-model)
    - [FLAN-T5 with GloVe Embeddings](#flan-t5-with-glove-embeddings)
  - [Prerequisites](#prerequisites)
  - [Installation and Setup](#installation-and-setup)
    - [Right now the output of question-answer pdf chatbot is not much good.  So i want to improve it. Please share your thoughts, suggessions, ideas.](#right-now-the-output-of-question-answer-pdf-chatbot-is-not-much-good--so-i-want-to-improve-it-please-share-your-thoughts-suggessions-ideas)
  - [Future Improvements](#future-improvements)
  - [Contributing](#contributing)

## Introduction

This repository contains a document-based chatbot application that uses two different architectures for question answering and document summarization:

1. **DistilBERT-based Model**: Utilizes the `distilbert-base-cased-distilled-squad` model for question answering on document content.
2. **FLAN-T5 with GloVe Embeddings**: Uses GloVe embeddings for context retrieval and the `flan-t5-small` model for generating answers or summaries based on the context.

The chatbot can answer questions related to the document content and summarize sections based on user queries.

## Features

- **Document Understanding**: Automatically extracts and processes text from documents (PDF format).
- **Question Answering**: Responds to user queries using relevant content from the document.
- **Contextual Understanding**: Uses GloVe embeddings to find the most relevant parts of the document based on the query.
- **Multiple Model Support**: Supports both DistilBERT and FLAN-T5 models for different types of tasks.

## Models Used

### DistilBERT-based Model
- **Model**: `distilbert-base-cased-distilled-squad`
- **Purpose**: Extractive question answering based on the document content.
- **Use Case**: Quick retrieval of specific answers from documents.

### FLAN-T5 with GloVe Embeddings
- **Model**: `flan-t5-small`
- **Embeddings**: GloVe embeddings for context retrieval.
- **Purpose**: Generates answers or summaries based on context identified using GloVe embeddings.
- **Use Case**: Provides more contextual and detailed responses.

## Prerequisites

- Python 3.8 or above
- PyTorch
- Transformers Library (`transformers`)
- PyMuPDF (`fitz`)
- NumPy
- NLTK
- scikit-learn

## Installation and Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/DevadattaP/document-chatbot.git
    cd document-chatbot
    ```

2. **Download and set up the models and embeddings:**
    - **DistilBERT Model:**
        - Download the `distilbert-base-cased-distilled-squad` model and place it in the appropriate directory.
    - **FLAN-T5 Model:**
        - Download the `flan-t5-small` model and place it in the appropriate directory.
    - **GloVe Embeddings:**
        - Download the GloVe embeddings (e.g., `glove.6B.300d.txt`) and place them in the appropriate directory.


### Right now the output of question-answer pdf chatbot is not much good.  So i want to improve it. Please share your thoughts, suggessions, ideas.

## Future Improvements
- Fine-tuning Models: Improve model accuracy by fine-tuning on domain-specific data.
- Better Context Retrieval: Enhance the context retrieval process using more advanced techniques like sentence-BERT or dense retrieval.
- Hybrid Model Approaches: Combine multiple models for different tasks (e.g., retrieval and generation).


## Contributing
Contributors are welcome !