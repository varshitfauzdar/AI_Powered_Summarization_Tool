# 📄 AI-Powered Research Paper Assistant

An intelligent NLP-based system that summarizes research papers and enables interactive question answering using a hybrid **RAG (Retrieval-Augmented Generation)** pipeline with **Section-Aware Retrieval**.

---

## 🧠 Project Overview

Reading research papers is time-consuming due to their length, complexity, and structured format.  
This project aims to:

- 📌 Generate concise summaries of research papers  
- 💬 Answer user queries based on paper content  
- 🧠 Improve retrieval using document structure awareness  

---

## 🚀 Key Features

- 📄 Upload any research paper (PDF)
- 📝 Automatic summarization
- 💬 Ask questions about the paper
- 🧠 Section-aware intelligent retrieval
- 🔍 Semantic chunking for better context understanding
- ⚡ Fast and interactive Streamlit UI

---

## 🧠 Core Technologies & Models

### 🔹 Embedding Model
- **Sentence-BERT (all-MiniLM-L6-v2)**
- Converts text into semantic embeddings
- Used for similarity search in RAG pipeline

---

### 🔹 Language Model (LLM)
- **FLAN-T5 (google/flan-t5-small)**
- Instruction-tuned transformer
- Used for:
  - Answer generation
  - Summarization

---

### 🔹 Vector Search
- **FAISS (Facebook AI Similarity Search)**
- Efficient similarity search for embeddings
- Enables fast retrieval of relevant content

---

## 🧱 System Architecture</br>
PDF Input</br>
↓</br>
Text Extraction (PyMuPDF)</br>
↓</br>
Text Cleaning & Preprocessing</br>
↓</br>
Section Detection (Abstract, Method, Results, etc.)</br>
↓</br>
Sentence Filtering & Semantic Chunking</br>
↓</br>
Embedding (SBERT)</br>
↓</br>
FAISS Vector Index</br>
↓</br>
Query Processing</br>
↓</br>
Section-Aware Retrieval ⭐</br>
↓</br>
LLM (FLAN-T5)</br>
↓</br>
Final Answer / Summary</br>


---

## 🔥 Key Innovations

Phase 1 → Basic RAG system

### ⭐ 1. Semantic Chunking
Instead of fixed-size chunks, sentences are grouped based on semantic similarity using SBERT embeddings.

✔ Preserves contextual meaning  
✔ Improves retrieval quality  

---

## 🚀 Phase 2 Upgrade: Hybrid Extractive + Abstractive Summarization 🔥

### 🧠 Motivation

In the initial version of our system, we used a **pure LLM-based summarization approach**, where entire sections of the research paper were directly passed to the language model (FLAN-T5) for summary generation.

However, this approach had several limitations:

- ❌ Repetition in generated summaries  
- ❌ Inclusion of noisy or irrelevant content (e.g., template text, metadata)  
- ❌ Lack of focus on key important sentences  
- ❌ Inefficient handling of long documents  

---

### 🔥 Proposed Improvement

To address these issues, we introduced a **Hybrid Summarization Approach** that combines:

- **Extractive Summarization (Sentence Ranking using SBERT)**  
- **Abstractive Summarization (FLAN-T5)**  

---

### ⚙️ How It Works</br>
Section Text</br>
↓</br>
Sentence Splitting</br>
↓</br>
Semantic Embedding (SBERT)</br>
↓</br>
Sentence Ranking (Cosine Similarity)</br>
↓</br>
Top-K Important Sentences Selected</br>
↓</br>
LLM (FLAN-T5)</br>
↓</br>
Final Summary</br>


---

### 🧩 Key Idea

Instead of sending the entire section to the LLM, we:

1. Extract meaningful sentences  
2. Rank them based on semantic importance  
3. Select top relevant sentences  
4. Pass only filtered content to the LLM  

---

### ✅ Benefits

- 🔥 Reduces repetition in summaries  
- 🎯 Focuses on important content only  
- 🧹 Removes noise from raw PDF extraction  
- 📄 Improves summary coherence and readability  
- ⚡ More efficient processing for long documents  

---

### ⚔️ Comparison with Previous Approach

| Feature | Pure LLM Approach | Hybrid Approach |
|--------|------------------|----------------|
| Input to LLM | Full section text | Filtered important sentences |
| Noise Handling | ❌ Poor | ✅ Strong |
| Repetition | ❌ High | ✅ Low |
| Control over content | ❌ None | ✅ High |
| Summary Quality | ⚠️ Moderate | 🔥 High |

---

### 🧠 Technical Implementation

- **Sentence Embeddings:** SBERT (`all-MiniLM-L6-v2`)  
- **Similarity Measure:** Cosine Similarity  
- **Ranking Strategy:** Sentence importance based on similarity matrix  
- **Generation Model:** FLAN-T5  

---

### 🎯 Outcome

This upgrade significantly improved the quality of summaries by ensuring that:

- Only meaningful and relevant content is processed  
- The LLM generates structured and concise summaries  
- Redundancy and hallucination are reduced  

---

### 🎤 Key Insight

> “Instead of summarizing everything, we first identify what is important, and then summarize it.”


---

### ⭐ 2. Section-Aware Retrieval (Main Contribution)
Traditional RAG retrieves from entire document.

We improve it by:
---
Query → Detect Section → Retrieve from that section
---


✔ Improves accuracy  
✔ Reduces irrelevant retrieval  
✔ Aligns with research paper structure  

---

### ⭐ 3. Intelligent Text Filtering
We remove:
- citations (e.g., *et al.*)
- short sentences
- noisy references

✔ Cleaner input  
✔ Better model performance  

---

### ⭐ 4. Hybrid RAG + Direct Section Retrieval
- Uses section-based filtering for precision  
- Uses RAG for semantic relevance  

---

## 📊 Pipeline Explanation

### 🔹 Summarization Pipeline
- Uses **Abstract + Introduction + Conclusion**
- Generates concise summary using FLAN-T5

---

### 🔹 Question Answering Pipeline
1. Detect relevant section
2. Split into meaningful sentences
3. Convert into embeddings (SBERT)
4. Retrieve top-k relevant sentences (FAISS)
5. Pass context to FLAN-T5
6. Generate answer

---

## 🖥️ User Interface

Built using **Streamlit**

Features:
- PDF Upload
- Summary Generation Button
- Interactive Q&A Input

---

## ⚙️ Installation & Setup

```bash
pip install streamlit pymupdf sentence-transformers faiss-cpu transformers torch

---

▶️ Run the Application

streamlit run app.py

📂 Project Structure
project/│├── app.py              # Streamlit UI├── rag_pipeline.py     # Core NLP pipeline├── README.md

🎯 Example Use Cases


📚 Students analyzing research papers


🔬 Researchers reviewing literature


🧠 NLP learning projects


📄 Automated document assistants



⚠️ Limitations


Limited context window of transformer models


May miss information if section detection is imperfect


Works best with well-structured research papers



🚀 Future Improvements


🔥 Fine-tuning on scientific datasets (e.g., SciBERT, Longformer)


📊 Add evaluation metrics (ROUGE, BLEU)


🌐 Deploy using FastAPI + Next.js


💬 Add chat history and memory


🧠 Improve section detection using ML models


📈 Confidence scoring system



🎤 Project Highlights (For Presentation)


Built a full RAG-based NLP system


Introduced Section-Aware Retrieval


Improved summarization quality using structured context


Implemented end-to-end pipeline from PDF → Answer



👨‍💻 Author
Developed as part of NLP project with focus on:


AI systems


LLM integration


real-world applications



⭐ Final Note
This project demonstrates how combining:


semantic understanding (SBERT)


retrieval (FAISS)


generation (FLAN-T5)


can create a powerful AI assistant for research understanding.
