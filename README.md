VectorSearch: A Multi-Modal E-commerce Search Engine

Team Members: 

$$Your Name$$

Tier Selection: Tier 2.5 (Advanced)

Justification: This project implements a core Tier 3 concept (a multi-modal search engine using CLIP and a vector database) but is scoped to an achievable Tier 2 level (functional prototype, 44k-image dataset, running in a free Colab environment). It is a "multi-component system" that is more advanced than a standard classification or detection project.

Problem Statement

Product discovery on e-commerce sites is inefficient. Searches rely on exact text keywords (e.g., "shirt"), which fails when users can't find the exact words (e.g., "button-down" vs. "formal") or want to "search by inspiration." This disconnect between user intent and search capability leads to customer frustration and lost sales.

Solution Overview

This project is a prototype search engine that understands the visual content and conceptual meaning of products, not just their text tags. The system allows users to search a 44,000-item fashion catalog using either natural language text (e.g., "a blue summer dress") or by uploading an image to find visually similar items.

Technical Approach

Technique: Cross-Modal Retrieval & Image Embedding

Model: CLIP (Contrastive Language–Image Pre-Training) from OpenAI.

Frameworks:

Hugging Face transformers: To load the pre-trained CLIP model.

FAISS (by Facebook AI): A highly efficient library for similarity search in the vector database.

PyTorch: As the backend for the CLIP model.

Dataset Plan

Source: Public Kaggle Dataset: "Fashion Product Images Dataset"

Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Size: ~44,400 product images.

Labels & Preparation: We will use the productDisplayName from the styles.csv file as the text data. A one-time script will be run to process all images through the CLIP model, generating vector embeddings. These embeddings will be stored in a FAISS index file.

Success Metrics

Metric Type

Metric

Target

Primary Metric

Retrieval Relevance (Qualitative)

For 10 test queries (5 text, 5 image), at least 4 out of 5 retrieved results are "highly relevant" to the query's intent.

Secondary Metric

Query Latency

< 1 second per query (demonstrating the speed of the FAISS index, excluding one-time model load).

Project Structure

ITAI 1378 Midterm_VectorSearch/
├── README.md                 <-- This file
├── requirements.txt          <-- Python packages (transformers, faiss-cpu, torch, etc.)
├── notebooks/
│   ├── 01_data_embedding.ipynb <-- Script to process all images and build the FAISS index
│   └── 02_search_demo.ipynb    <-- The final interactive demo notebook
├── data/
│   └── README.md             <-- Info on how to download the Kaggle dataset
├── docs/
│   └── proposal.pdf          <-- A PDF export of the presentation slides
└── .gitignore


Week-by-Week Plan

Week

Date (2025)

Task

Milestone

10

Oct 30

Setup & Data Prep

Dataset downloaded. GitHub repo & requirements.txt created.

11

Nov 6

Embedding Pipeline

Write script to load CLIP, process all 44k images, and save embeddings.

12

Nov 13

Vector DB & Text Search

Load embeddings into FAISS. Build & test the core text-to-image search.

13

Nov 20

Multi-Modal Demo

Implement image-to-image search. Build interactive demo (Gradio/ipywidgets).

14

Nov 27

Finalize & Document

Clean code, record video, finalize README.md & presentation.

15

Dec 4

Present Project

🎉 Presentation day.

Risks & Mitigation Table

Risk

Probability

Mitigation Plan

Colab Time/Compute Limits

Medium

The 44k image embedding process may time out. Plan B: Process images in smaller batches (e.g., 5k at a time) and merge the FAISS indexes.

Low Search Relevance

Medium

The default CLIP model may not understand fashion concepts well. Plan B: Experiment with a different pre-trained CLIP model variant from Hugging Face.

Demo UI is Buggy

Low

The Gradio or ipywidgets demo might fail. Plan B: The demo will revert to simple Python function calls in the Colab notebook, printing the query and displaying the output images.

Resources Needed

Compute: Google Colab (Free Tier). The free GPU is sufficient for inference with a pre-trained model.

Frameworks: PyTorch, Hugging Face transformers, FAISS, Gradio (optional).

Estimated Cost: $0.

🧾 AI Usage Log

$$2025-10-30$$

 Used Gemini to brainstorm Tier 2/3 project ideas, assess the feasibility of a generative AI project, and refine the scope to the "Lite" Multi-Modal Search concept.

$$2025-10-30$$

 Used Gemini to draft the 10-slide proposal presentation and this README.md file based on the course requirements and project idea.

... (This log will be updated throughout the project) ...
