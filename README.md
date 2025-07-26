# 📚 Bengali RAG QA System (PDF ➝ OCR ➝ FAISS ➝ mBERT)

This project is a Bengali-English bilingual **Retrieval-Augmented Generation (RAG)** system that answers natural language questions from a scanned textbook PDF. It uses OCR to extract text, semantic chunking + embedding, and FAISS-based vector search to find relevant content, and a multilingual transformer model for answering user questions.

> ✅ Fully local pipeline — No paid API needed  
> 🧠 Powered by Tesseract OCR, FAISS, and multilingual SBERT + XLM-RoBERTa  
> 🌐 Supports both Bangla and English queries

---

## 🔧 Setup Guide

### 🛠 Requirements

- Ubuntu or Linux (tested on Kaggle/Colab)
- Python 3.8+
- Tesseract OCR with Bengali support

### 🐍 Install Dependencies

```bash
# Linux packages
!apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben

# Python packages
!pip install pytesseract pdf2image sentence-transformers faiss-cpu transformers nltk --quiet
```

---

## 📁 Input PDF

```python
pdf_path = "/kaggle/input/10mins/HSC26-Bangla1st-Paper.pdf"
```

This system was tested on the **HSC26 Bangla 1st Paper** PDF.

---

## 🧰 Tools, Libraries & Packages

| Tool/Library | Purpose |
|--------------|---------|
| `pytesseract` | OCR to extract Bangla text from scanned images |
| `pdf2image` | Convert PDF pages into high-resolution images |
| `nltk` | Sentence tokenization |
| `sentence-transformers` | Multilingual embedding of text chunks |
| `faiss-cpu` | Efficient semantic similarity search |
| `transformers` | Multilingual QA model (`xlm-roberta-large-squad2`) |

---

## 🧠 System Pipeline

### 1. **OCR Extraction**

```python
text = pytesseract.image_to_string(page, lang='ben')
```

- **Library used:** `pytesseract` with `tesseract-ocr-ben`
- **Why:** Open-source OCR that supports Bengali script well.
- **Formatting Challenges:** Yes, OCR often introduces unwanted line breaks, punctuation errors, and inconsistent character spacing. Unicode normalization helps reduce noise.

---

### 2. **Chunking Strategy**

```python
sentences = sent_tokenize(full_text)
chunks = [' '.join(sentences[i:i + 5]) for i in range(0, len(sentences), 5)]
```

- **Type:** Sentence-based, fixed-size (5 sentences per chunk)
- **Why it works:** Sentence grouping maintains semantic flow while avoiding overly large inputs. It's also resilient to OCR noise and sentence fragments.

---

### 3. **Text Embedding**

```python
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```

- **Why this model:**  
  - Lightweight and fast (MiniLM architecture)  
  - Supports **100+ languages** including Bangla  
  - Trained on paraphrase tasks, ideal for semantic similarity  
- **How it works:**  
  Transforms each chunk into a high-dimensional vector encoding its semantic meaning, capturing both syntax and context.

---

### 4. **Vector Similarity Search**

```python
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))
```

- **Comparison Method:** L2 (Euclidean) distance
- **Why FAISS:**  
  - Fast and memory-efficient
  - Scales to thousands of chunks
- **Why it works:**  
  Closest vectors represent most semantically similar chunks to the query.

---

### 5. **Question Answering Inference**

```python
qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")
```

- **Why this model:**  
  - Robust multilingual transformer  
  - Fine-tuned on SQuAD2, capable of handling answerable/unanswerable questions  
- **Process:**  
  - Retrieves top-k chunks  
  - Concatenates them as QA context  
  - Uses transformer to extract answer span

---

## 💬 Sample Queries & Outputs

| Query (Bangla)                          | Output                                      |
|----------------------------------------|---------------------------------------------|
| `কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`                 | `মামাকে`                        |
| ` বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?`         | `পনেরো`                                     |


---

## 🧪 Evaluation (Planned but Not Implemented)

### Metrics (Optional for Future Implementation)

| Metric     | Description |
|------------|-------------|
| EM (Exact Match) | Checks if model output exactly matches the ground truth |
| F1 Score   | Token-level overlap between prediction and ground truth |
| Top-k Retrieval Accuracy | Whether relevant chunk was among top-k |

---

## 📡 API Documentation (Optional)

```python
def answer_query_with_qa_model(query: str, top_k: int = 5) -> str:
    ...
    return result['answer']
```

---

## ❓ Methodological Q&A

See full answers in the markdown above (OCR choice, chunking strategy, embedding model, retrieval design, evaluation reasoning, etc.)

---

## 📌 Future Work

- [ ] Add mT5 or IndicBERT for better Bangla performance
- [ ] Implement evaluation metrics (F1, EM)
- [ ] Deploy with Gradio or Streamlit UI
- [ ] Extend to more textbooks or subjects

---

## 📜 License

MIT License. Not affiliated with Bangladesh NCTB or any publisher.
