# VectorSearch: Multi-Modal Fashion Search Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/Model-CLIP-green.svg)](https://github.com/openai/CLIP)
[![FAISS](https://img.shields.io/badge/Index-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A multi-modal search engine using CLIP and a vector database to power both text-to-image and image-to-image search across a 44,000-item fashion catalog.**

**Course:** ITAI 1378 - Computer Vision  
**Author:** Alhassane Samassekou  
**Institution:** Houston City College  
**Project Type:** Final Project

---

## Demo Video link
https://drive.google.com/file/d/1xLhumZZUQtEVR3VZwlSVyYf30DyU7Ku6/view?usp=sharing

## Project Overview
VectorSearch is a multi-modal search engine that enables users to search a fashion e-commerce catalog using both **natural language text queries** and **image uploads**. Built using CLIP (Contrastive Language-Image Pre-training) and FAISS vector database, this system allows users to find products through semantic search rather than relying on exact keyword matches.
### Key Capabilities
- **Natural Language Search**: Query using descriptive phrases like "blue summer dress"
- **Image-Based Search**: Upload an image to find visually similar items
- **Fast Retrieval**: Sub-second query latency across 44,000+ products
- **Semantic Understanding**: Goes beyond exact keyword matching
---
## The Problem
### Real-World Challenge
Product discovery on e-commerce sites is fundamentally inefficient:
- **Vocabulary Gap**: Traditional search relies on exact text keywords but fails when users can't describe items precisely (e.g., "formal shirt" vs. "button-down oxford")
- **Visual Complexity**: Complex visual features are difficult to describe accurately with text alone
- **Limited Discovery**: Impossible to "search by inspiration" by uploading a photo of a style you saw
### Who This Affects
- **E-commerce businesses** - Lost sales due to poor search experience
- **Online shoppers** - Frustration from inability to find desired products
- **Digital marketers** - Reduced conversion rates and product discovery
### Why It Matters
Poor search leads directly to customer frustration and lost sales. This limits product discovery opportunities and reduces overall conversion rates, impacting both customer satisfaction and business revenue.
---
## The Solution
VectorSearch bridges the vocabulary gap by mapping both images and text into the same embedding space, enabling true multi-modal search capabilities.
### How It Works
```
Input: Text Query ("red dress") OR Image Query (upload.jpg)
↓
Step 1: CLIP Encoder
(Converts text or image query into 512-dimensional vector)
↓
Step 2: FAISS Vector Database
(Searches pre-computed 44k-image index for nearest neighbors)
↓
Output: Top 5 Similar Products
(Displays the most relevant retrieved images)
```
### Technology Foundation
- **CLIP Model**: Maps both images and text to the same embedding space, enabling cross-modal retrieval
- **FAISS Vector Database**: Enables efficient similarity search across embeddings with sub-millisecond latency
- **PyTorch & Transformers**: Framework for model inference and deployment
---
## System Architecture
VectorSearch employs a retrieve-and-rank pipeline for efficient multi-modal search:
```
User Query     
(Text/Image)   
CLIP Encoder   
(512-dim)      
L2 Normalize   
FAISS Index    
(44k vectors)  
Top-K         
Results       
```
### Pipeline Stages
1. **Query Encoding**: CLIP converts text or image input into a 512-dimensional vector embedding
2. **Normalization**: L2 normalization enables cosine similarity search
3. **Vector Search**: FAISS efficiently retrieves the K nearest neighbors from the pre-computed index
4. **Result Display**: Top-K most similar products are returned to the user
---
## Dataset
**Source**: [Fashion Product Images Dataset (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
### Specifications
| Attribute | Details |
|-----------|---------|
| **Total Images** | ~44,400 product images |
| **Metadata** | styles.csv with product descriptions |
| **Labels** | ProductDisplayName (e.g., "Blue T-Shirt") paired with images |
| **Categories** | Apparel, Footwear, Accessories, Personal Care, and more |
| **Format** | High-resolution product photos with white backgrounds |
### Data Preparation Pipeline
1. **Image Processing**: Load all ~44K images through pre-trained CLIP model
2. **Embedding Generation**: Generate 512-dimensional vector embeddings for each image
3. **Index Building**: Store embeddings in FAISS index for efficient retrieval
4. **Persistence**: Save index to disk for fast loading during inference
---
## Technical Stack
### Frameworks & Libraries
| Technology | Purpose | Version |
|------------|---------|---------|
| **PyTorch** | Deep learning framework | 2.0+ |
| **Hugging Face Transformers** | CLIP model implementation | 4.30+ |
| **FAISS** | Efficient vector similarity search | 1.7.4+ |
| **Gradio** | Interactive web demo UI | 4.0+ (optional) |
| **Pandas** | Data processing and management | 2.0+ |
| **NumPy** | Numerical computations | 1.24+ |
### Compute Requirements
- **Platform**: Google Colab (Free Tier compatible)
- **GPU**: Free tier GPU sufficient for inference  
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for dataset and embeddings
- **Cost**: $0 - Designed for free-tier resources
### Why These Technologies?
- **CLIP**: Industry standard for mapping images and text to the same embedding space
- **FAISS**: Highly efficient, lightweight library perfect for prototyping in Colab environments
- **PyTorch**: Most widely used deep learning framework with excellent community support
- **Gradio**: Enables rapid prototyping of interactive demos with minimal code
---
## Installation
### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- 8GB RAM minimum
- 5GB free disk space
### Step 1: Clone the Repository
```bash
git clone https://github.com/asamassekou10/ITAI-1378-FINAL_VectorSearch.git
cd ITAI-1378-FINAL_VectorSearch
```
### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Or using conda
conda create -n vectorsearch python=3.10
conda activate vectorsearch
```
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Download Dataset
You'll need a Kaggle API key to download the Fashion Product Images dataset.
1. Get your Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account)
2. Place `kaggle.json` in the project root directory
```bash
# Install Kaggle CLI
pip install kaggle
# Download dataset
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip
```
---
## Usage
### Quick Start - Notebook Workflow
The project consists of three main notebooks that should be run in sequence:
#### 1. Setup and Exploration
```bash
jupyter notebook 01_setup_and_exploration_Alhassane_Samassekou.ipynb
```
**Purpose**: 
- Load and explore the Fashion Product dataset
- Analyze data distribution and quality
- Prepare metadata for embedding generation
#### 2. Embedding Pipeline
```bash
jupyter notebook 02_embedding_pipeline_Alhassane_Samassekou.ipynb
```
**Purpose**:
- Generate CLIP embeddings for all product images
- Create and save the FAISS index (`image_embeddings.npy`)
- Validate embedding quality
**⏱ Processing Time**: ~30-60 minutes for 44k images on CPU
#### 3. Search Demo
```bash
jupyter notebook 03_search_demo_Alhassane_Samassekou_Professional.ipynb
```
**Purpose**:
- Load pre-computed embeddings and FAISS index
- Initialize the CLIP model for query encoding
- Launch the interactive Gradio web interface
### Alternative: Command-Line Usage
If you prefer command-line scripts over notebooks:
#### Generate Image Embeddings (One-Time Setup)
```bash
python src/generate_embeddings.py --data_path ./data/images --output_path ./embeddings/
```
#### Build FAISS Index
```bash
python src/build_index.py --embeddings_path ./embeddings/image_embeddings.npy
```
#### Run Search Queries
**Text-to-Image Search**:
```bash
python src/search.py --query "blue summer dress" --top_k 5
```
**Image-to-Image Search**:
```bash
python src/search.py --image_path ./query_image.jpg --top_k 5
```
#### Launch Demo UI
```bash
python src/demo.py
```
The Gradio interface will launch at `http://localhost:7860`
### Using the Search Interface
**Text Search:**
1. Enter a description in the "Text Query" field (e.g., "red formal dress")
2. Select a category filter (optional)
3. Click " Search"
**Image Search:**
1. Upload an image using the "Image Query" panel
2. Select a category filter (optional)
3. Click " Search"
**Note**: Image queries take precedence over text when both are provided.
---
## Project Structure
```
vectorsearch/
README.md
notebooks/
01_setup_and_exploration_Alhassane_Samassekou.ipynb
02_embedding_pipeline_Alhassane_Samassekou.ipynb
03_search_demo_Alhassane_Samassekou_Professional.ipynb
docs/
Présentation.pdf
AI_usage_log.md

```
---

## Challenges & Solutions
### Challenge 1: Colab Compute/Time Limits
**Problem**: 44K image embedding process may exceed Colab session limits  
**Solution Implemented**: 
- Batch processing with checkpoint saving every 5,000 images
- Reduced batch size to 32 for memory efficiency
- Added progress tracking with tqdm
**Backup Plan**: Process images in smaller chunks (5K at a time) or reduce dataset to 10K images
### Challenge 2: Search Relevance
**Problem**: Retrieved results might not match query intent  
**Solution Implemented**:
- Used pre-trained CLIP ViT-B/32 (proven for fashion domain)
- Implemented proper L2 normalization for cosine similarity
- Added category filtering for targeted results
**Backup Plan**: Experiment with ViT-L/14 or domain-specific CLIP variants from HuggingFace
### Challenge 3: Demo UI Stability
**Problem**: Gradio interface might be buggy or crash  
**Solution Implemented**:
- Comprehensive error handling with try-except blocks
- Graceful degradation for missing images
- Input validation for empty queries
**Backup Plan**: Revert to simple Python function calls in Colab notebook if Gradio fails
---
## Learning Objectives
This project demonstrates mastery of:
1. **Multi-Modal Machine Learning**
- Understanding CLIP architecture and pre-training methodology
- Cross-modal embedding alignment between text and images
2. **Vector Embeddings & Similarity Search**
- Generating high-dimensional vector representations
- Efficient similarity search with FAISS indexing
3. **System Design & Optimization**
- Batch processing for large-scale data
- GPU acceleration and memory management
- Building scalable search pipelines
4. **Real-World Applications**
- Solving practical e-commerce challenges
- Building production-ready prototypes
- User interface design for AI systems
5. **Software Engineering**
- Version control with Git/GitHub
- Documentation and code organization
- Reproducible research practices
---
## Technical Implementation
### 1. Embedding Generation
```python
# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Generate image embeddings
for img_path in image_paths:
image = Image.open(img_path)
inputs = processor(images=image, return_tensors="pt")
embedding = model.get_image_features(**inputs)
embeddings.append(embedding.detach().numpy())
```
### 2. FAISS Index Creation
```python
import faiss
# Initialize FAISS index for Inner Product similarity
dimension = 512
index = faiss.IndexFlatIP(dimension)
# Add normalized embeddings
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings)
```
### 3. Search Function
```python
def search(query, category_filter="All", k=5):
# Encode query (text or image)
if isinstance(query, str):
inputs = processor(text=[query], return_tensors="pt")
vector = model.get_text_features(**inputs)
else:
inputs = processor(images=query, return_tensors="pt")
vector = model.get_image_features(**inputs)
# Normalize and search
vector = vector / vector.norm(p=2, dim=-1, keepdim=True)
vector = vector.cpu().detach().numpy()
# Oversample for filtering
search_k = k * 10 if category_filter != "All" else k
distances, indices = index.search(vector, search_k)
# Filter by category and format results
results = []
for idx, score in zip(indices[0], distances[0]):
item = df.iloc[idx]
if category_filter == "All" or item['masterCategory'] == category_filter:
results.append({
'image': item['image_path'],
'caption': item['productDisplayName'],
'score': score
})
if len(results) >= k:
break
return results
```
### 4. Oversampling Strategy
When category filters are applied, the system uses **10x oversampling** to ensure sufficient results:
```
Without Filter: Retrieve top 5 → Return 5 results
With Filter:    Retrieve top 50 → Filter by category → Return top 5 matches
```
This approach maintains high relevance without requiring category-specific indices.
---
## Performance Metrics
| Metric | Value | Details |
|--------|-------|---------|
| **Dataset Size** | 44,419 products | Enterprise-scale catalog |
| **Vector Dimension** | 512 | Balanced accuracy/speed |
| **Query Latency** | < 0.1 seconds | Real-time performance |
| **Index Type** | Flat (Exact) | 100% accuracy on retrieval |
| **Search Modalities** | 2 (Text + Image) | Multi-modal queries |
| **Categories** | 7 types | Apparel, Footwear, Accessories, etc. |
| **Embedding Model** | CLIP ViT-B/32 | 400M parameters |
| **GPU Memory** | ~2GB | Inference requirements |
### Validation Results
**Text-to-Image Search:**
- Semantic color understanding (e.g., "red" retrieves red items)
- Style matching (e.g., "formal shirt" finds dress shirts)
- Cross-category queries work effectively
**Image-to-Image Search:**
- Visual similarity matching (e.g., watches find similar watches)
- Pattern recognition across product types
- Robust to image quality variations
---
## Future Enhancements
1. **Hybrid Search**
- Combine vector similarity with BM25 keyword search
- Implement weighted fusion of semantic and lexical matching
2. **Re-Ranking**
- Add cross-encoder stage for top-K refinement
- Improve result ordering accuracy
3. **Fine-Tuning**
- Train CLIP on fashion-specific data
- Better understanding of domain terminology


</div>
