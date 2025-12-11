# AI Usage Log - VectorSearch Pro Project

**Project:** VectorSearch Pro - Multi-Modal Fashion Search Engine  
**Course:** ITAI 1378 - Computer Vision  
**Student:** Alhassane Samassekou  

---

## Purpose

This log documents AI assistance used during development, demonstrating transparent and responsible AI use while maintaining academic integrity.

---

## Entry 1: Project Architecture Design

**AI Tool:** Claude (Anthropic)  
**Purpose:** Designing system architecture and pipeline

**What I Asked:**
"Help me design a fashion search engine using CLIP and FAISS that handles 44k+ products with text/image queries and category filtering."

**AI Suggestion:**
Two-stage pipeline: (1) CLIP encoding to 512-dim vectors, (2) FAISS IndexFlatIP for similarity search. Use L2 normalization and 10x oversampling for category filters.

**How I Used It:**
Implemented the pipeline in notebook 3, used IndexFlatIP for exact search, added oversampling logic for filtered queries.

**What I Learned:**
IndexFlatIP gives exact results (good for proof-of-concept), oversampling prevents empty results after filtering.

---

## Entry 2: CLIP Model Selection

**AI Tool:** Claude (Anthropic)  
**Purpose:** Choosing the right CLIP model variant

**What I Asked:**
"What's the difference between CLIP ViT-B/32 and ViT-L/14? Which for fashion search on Colab with limited GPU?"

**AI Suggestion:**
ViT-B/32 (400M params, 512-dim) is better than ViT-L/14 (1B params, 768-dim) for Colab due to memory constraints. Use `openai/clip-vit-base-patch32` from HuggingFace.

**How I Used It:**
Selected ViT-B/32 for the project, achieved <0.3s query latency with 512-dim embeddings.

**What I Learned:**
Model selection balances accuracy, speed, and resources. ViT-B/32 sufficient for fashion without fine-tuning.

---

## Entry 3: Embedding Generation Optimization

**AI Tool:** Claude + GitHub Copilot  
**Purpose:** Optimizing batch processing for 44k images

**What I Asked:**
"Generating CLIP embeddings for 44,419 images taking 3+ hours. How to optimize on Colab T4 GPU?"

**AI Suggestion:**
Use batch processing, GPU acceleration with `.to(device)`, `torch.no_grad()`, tqdm progress bars, and incremental saving.

**How I Used It:**
Implemented batch_size=32, added no_grad() context, used tqdm for tracking. Reduced time from 3+ hours to ~45 minutes.

**What I Learned:**
Batch processing improves GPU utilization, no_grad() saves memory during inference, incremental saves prevent data loss.

---

## Entry 4: FAISS Index Configuration

**AI Tool:** Claude (Anthropic)  
**Purpose:** Understanding FAISS index types

**What I Asked:**
"Explain IndexFlatIP vs IndexFlatL2 vs IndexIVFFlat. Which for 44k vectors, 512-dim, exact search?"

**AI Suggestion:**
IndexFlatIP for normalized vectors (cosine similarity), IndexFlatL2 for unnormalized, IndexIVFFlat for approximate search. Use IndexFlatIP with L2 normalization.

**How I Used It:**
Implemented IndexFlatIP with L2 normalization, achieved exact search with <0.1s latency.

**What I Learned:**
Flat indices work for <1M items. Normalization crucial for correct similarity. Would need IVF/HNSW for larger scales.

---

## Entry 5: Gradio Interface Design

**AI Tool:** Claude (Anthropic)  
**Purpose:** Creating user-friendly web interface

**What I Asked:**
"Design Gradio interface for fashion search: text input, image upload, category dropdown, gallery results, latency display."

**AI Suggestion:**
Use `gr.Blocks()` with two-column layout (inputs left, results right), `gr.Gallery()` for results, `gr.Label()` for latency, Soft theme.

**How I Used It:**
Implemented two-column layout, gallery with 2 columns, real-time latency display, modified captions to show product name and score.

**What I Learned:**
Gradio Blocks give more control than Interface. Gallery handles image display automatically. Real-time metrics improve UX.

---

## Entry 6: Category Filter Implementation

**AI Tool:** Claude (Anthropic)  
**Purpose:** Implementing post-retrieval filtering

**What I Asked:**
"If I retrieve top 5 then filter by category, I might get 0 results. How to handle?"

**AI Suggestion:**
Use oversampling: retrieve k*10 candidates (e.g., 50), filter by category, return top k matches (e.g., 5).

**How I Used It:**
Implemented `search_k = k * 10 if category_filter != "All" else k`, added filtering loop, merged styles.csv for categories.

**What I Learned:**
Oversampling solves post-filtering issue. Alternative would be separate indices per category (more complex). 10x provides good balance.

---

## Entry 7: Error Handling

**AI Tool:** Claude (Anthropic)  
**Purpose:** Handling edge cases for robustness

**What I Asked:**
"What edge cases to handle? Users might submit empty queries, invalid images, filters with no results, or face network issues."

**AI Suggestion:**
Handle empty queries, image processing errors, model loading failures, zero results, CUDA OOM. Add try-except blocks with user-friendly messages.

**How I Used It:**
Added validation in `run_search()` for empty queries, wrapped image loading in try-except, added device detection with CPU fallback.

**What I Learned:**
Error handling improves UX. Graceful degradation better than crashes. Clear error messages are important.

---

## Entry 8: Documentation

**AI Tool:** Claude (Anthropic)  
**Purpose:** Creating clear documentation and comments

**What I Asked:**
"Help write documentation for notebooks: section headers, inline comments, markdown cells, professional academic style for course submission."

**AI Suggestion:**
Use markdown for section context, comment complex code only, keep simple code uncommented, add executive summary and conclusions.

**How I Used It:**
Added comprehensive markdown cells, documented complex operations (normalization, oversampling), kept simple code uncommented, added intro and conclusion.

**What I Learned:**
Documentation as important as code. Markdown explains "why", code explains "how". Academic projects need different docs than production.

---

## Entry 9: Performance Benchmarking

**AI Tool:** Claude (Anthropic)  
**Purpose:** Measuring system performance

**What I Asked:**
"How to benchmark search engine? What metrics matter for semantic search to show production-ready performance?"

**AI Suggestion:**
Track query latency (<1s target), index build time, memory usage, retrieval quality, queries per second. Use `time.time()` for timing.

**How I Used It:**
Added timing in search function, displayed latency in Gradio UI, documented <0.1s per query, created performance metrics table.

**What I Learned:**
Performance metrics validate scale. Real-time latency builds user confidence. Sub-second response critical for UX.

---

## Entry 10: GitHub Documentation

**AI Tool:** Claude (Anthropic)  
**Purpose:** Creating professional GitHub repository documentation

**What I Asked:**
"Create professional presentation and GitHub docs: README, setup guide, contribution guidelines for academic submission and portfolio."

**AI Suggestion:**
Comprehensive README with badges, installation, usage examples; SETUP.md with OS-specific instructions; CONTRIBUTING.md for collaboration; requirements.txt; .gitignore for large files.

**How I Used It:**
Adapted README to my implementation, created presentation slides with consistent branding, customized setup for my environment, added screenshots and demos.

**What I Learned:**
Professional documentation enhances project presentation. README is first impression. Good docs make projects accessible. Transparent AI use shows integrity.

---

## Summary

**Total Entries:** 10  
**AI Tools Used:** Claude (Anthropic), GitHub Copilot

### Assistance Categories:
- **Architecture & Design** (Entries 1-2): System design, model selection
- **Implementation** (Entries 3-6): Optimization, indexing, UI, filtering  
- **Quality Assurance** (Entries 7, 9): Error handling, benchmarking
- **Documentation** (Entries 8, 10): Comments, README, presentation

### What I Coded Myself:
- All three Jupyter notebooks from scratch
- Integration of CLIP with FAISS
- Custom search function with oversampling
- Gradio interface and event handlers
- Category filtering and data merging
- Testing and validation of results
- Performance optimization implementation
