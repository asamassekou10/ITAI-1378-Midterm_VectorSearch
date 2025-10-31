# VectorSearch: A Multi-Modal E-Commerce Search Engine

**ITAI 1378 - Computer Vision Midterm Project**  
**Author:** Alhassane Samassekou

## 🎯 Project Overview

VectorSearch is a "lite" multi-modal search engine that enables users to search a fashion e-commerce catalog using both natural language text queries and image uploads. Built using CLIP (Contrastive Language-Image Pre-training) and FAISS vector database, this system allows users to find products through semantic search rather than relying on exact keyword matches.

### One-Sentence Description
A multi-modal search engine using CLIP and a vector database to power both text-to-image and image-to-image search across a 44,000-item fashion catalog.

## 🔍 The Problem

**Real-World Problem:**
- Product discovery on e-commerce sites is inefficient
- Traditional search relies on exact text keywords
- Fails when users can't describe items precisely (e.g., "formal shirt" vs. "button-down")
- Impossible to "search by inspiration" (uploading a photo of a style you saw)

**Who Cares?**
- E-commerce businesses
- Digital marketers
- Online shoppers

**Why It Matters:**
- Poor search leads directly to customer frustration and lost sales
- Limits product discovery opportunities
- Reduces conversion rates

## 💡 The Solution

### System Capabilities

The system allows users to search a 44,000-item fashion catalog using:

1. **Natural Language Text** - Search using descriptive phrases like "a blue summer dress"
2. **Image Upload** - Upload an image to find visually similar items

### How It Works

The system uses:
- **CLIP Model**: Maps both images and text to the same embedding space
- **FAISS Vector Database**: Enables efficient similarity search across embeddings
- **PyTorch & Hugging Face Transformers**: Framework for model inference

## 🏗️ System Architecture

```
Input: Text Query ("red dress") OR Image Query (upload.jpg)
           ↓
Processing Step 1: CLIP Encoder
    (Converts text or image query into vector embedding)
           ↓
Processing Step 2: FAISS Vector Database
    (Searches pre-computed 44k-image index for 5 "nearest neighbors")
           ↓
Output: Top 5 Similar Products
    (Displays the 5 retrieved images)
```

## 📊 Dataset

**Source:** Fashion Product Images Dataset (Kaggle)

**Specifications:**
- **Size:** ~44,400 product images
- **Metadata:** styles.csv file with text descriptions
- **Labels:** ProductDisplayName (e.g., "Blue T-Shirt") paired with corresponding images

**Data Preparation Pipeline:**
1. Process all ~44K images through pre-trained CLIP model
2. Generate vector embeddings for each image
3. Store embeddings in FAISS index
4. Save index to disk for efficient querying

## 🛠️ Technical Stack

### Frameworks & Libraries
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - CLIP model implementation
- **FAISS** - Efficient vector similarity search
- **Gradio** (optional) - Demo UI

### Compute Requirements
- **Platform:** Google Colab (Free Tier)
- **GPU:** Free tier GPU sufficient for inference
- **Cost:** $0 (designed for free-tier resources)

### Justification
- **CLIP:** Industry standard for mapping images and text to the same embedding space
- **FAISS:** Highly efficient, lightweight library perfect for prototyping in Colab environments

## 📈 Success Metrics

### Primary Metric: Retrieval Relevance (Qualitative)
- **Method:** Manually score top-5 results for 10 test queries (5 text, 5 image)
- **Target:** At least 4 out of 5 retrieved images are "highly relevant" to query intent

### Secondary Metric: Query Latency
- **Method:** Measure time from query submission to results
- **Target:** < 1 second per query (demonstrating FAISS index speed)

## 📅 Development Timeline

### Week 10 (Oct 30) - Setup & Data Prep
**Milestone:** Dataset ready, GitHub repo & requirements.txt created

### Week 11 (Nov 6) - Embedding Pipeline
**Milestone:** image_embeddings.npy file created

### Week 12 (Nov 13) - Vector DB & Text Search
**Milestone:** Text-to-image search is functional

### Week 13 (Nov 20) - Multi-Modal Demo
**Milestone:** Image-to-image search added & demo ready

### Week 14 (Nov 27) - Finalize & Document
**Milestone:** All components done, README.md finalized

### Week 15 (Dec 4) - Present Project
**Milestone:** Presentation day

## ⚠️ Challenges & Backup Plans

### Challenge 1: Colab Compute/Time Limits
**Problem:** 44K image embedding process may exceed Colab limits  
**Plan B:** Process images in smaller batches (5K at a time); reduce scope to 10K-image dataset if necessary

### Challenge 2: Low Search Relevance
**Problem:** Results don't match query intent  
**Plan B:** Experiment with different pre-trained CLIP model variants from Hugging Face

### Challenge 3: Demo UI Issues
**Problem:** Gradio interface is buggy  
**Plan B:** Revert to simple Python function calls in Colab notebook

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision
pip install transformers
pip install faiss-cpu  # or faiss-gpu if available
pip install pillow
pip install pandas
pip install gradio  # optional, for UI
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/vectorsearch.git
cd vectorsearch

# Install requirements
pip install -r requirements.txt

# Download the dataset
# [Instructions for downloading Fashion Product Images Dataset from Kaggle]
```

### Usage

#### 1. Generate Image Embeddings (One-Time Setup)
```python
python generate_embeddings.py --data_path ./data/images --output_path ./embeddings/
```

#### 2. Build FAISS Index
```python
python build_index.py --embeddings_path ./embeddings/image_embeddings.npy
```

#### 3. Run Search Queries

**Text-to-Image Search:**
```python
python search.py --query "blue summer dress" --top_k 5
```

**Image-to-Image Search:**
```python
python search.py --image_path ./query_image.jpg --top_k 5
```

#### 4. Launch Demo UI (Optional)
```python
python demo.py
```

## 📁 Project Structure

```
vectorsearch/
├── README.md
├── requirements.txt
├── data/
│   ├── images/
│   └── styles.csv
├── embeddings/
│   └── image_embeddings.npy
├── models/
│   └── faiss_index.bin
├── src/
│   ├── generate_embeddings.py
│   ├── build_index.py
│   ├── search.py
│   └── demo.py
├── notebooks/
│   └── exploration.ipynb
└── docs/
    └── project_proposal.pdf
```

## 🎓 Learning Objectives

This project demonstrates:
- Multi-modal machine learning with CLIP
- Vector embeddings and similarity search
- Efficient indexing with FAISS
- Real-world computer vision applications
- Building scalable search systems

## 📝 License

This project is created for educational purposes as part of ITAI 1378 Computer Vision course.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Facebook Research for FAISS
- Kaggle for the Fashion Product Images Dataset
- Hugging Face for model implementations

## 📧 Contact

**Alhassane Samassekou**  
ITAI 1378 - Computer Vision  
alhassane.samassekou@gmail.com
---

*Built with a focus on innovation balanced with environmental sustainability, using free-tier resources to minimize computational costs.*
