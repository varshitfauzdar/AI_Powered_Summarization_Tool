# rag_pipeline.py

import fitz
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# LOAD MODELS (GLOBAL)
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# -------------------------------
# PDF PROCESSING
# -------------------------------
def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return text


# -------------------------------
# TEXT PROCESSING
# -------------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if len(s) < 80:
            continue
        if "et al" in s.lower():
            continue
        if "[" in s and "]" in s:
            continue
        if s and s[0].isdigit():
            continue

        cleaned.append(s)

    return cleaned


# -------------------------------
# SECTION DETECTION
# -------------------------------
def detect_sections(text):
    sections = {
        "abstract": "",
        "introduction": "",
        "method": "",
        "results": "",
        "conclusion": ""
    }

    current = "introduction"

    for line in text.split("\n"):
        l = line.lower()

        if "abstract" in l:
            current = "abstract"
        elif "introduction" in l:
            current = "introduction"
        elif "method" in l:
            current = "method"
        elif "result" in l:
            current = "results"
        elif "conclusion" in l:
            current = "conclusion"

        sections[current] += " " + line

    return sections


def get_section_from_query(query):
    q = query.lower()

    if "method" in q:
        return "method"
    elif "result" in q:
        return "results"
    elif "conclusion" in q:
        return "conclusion"
    elif "abstract" in q:
        return "abstract"
    else:
        return "introduction"


# -------------------------------
# RAG RETRIEVAL
# -------------------------------
def retrieve_context(query, sections):
    target_section = get_section_from_query(query)
    section_text = sections[target_section]

    sentences = split_sentences(section_text)

    if len(sentences) == 0:
        return ""

    embeddings = embed_model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k=min(5, len(sentences)))

    retrieved = [sentences[i] for i in I[0]]

    return " ".join(retrieved[:4])


# -------------------------------
# ANSWER GENERATION
# -------------------------------
def generate_answer(query, context):
    prompt = f"""
    You are an expert AI assistant for research papers.

    Answer clearly and completely using the context.

    Question: {query}

    Context:
    {context}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_length=300)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# SUMMARY GENERATION
# -------------------------------
def generate_summary(sections):
    context = (
        sections["abstract"] +
        sections["introduction"] +
        sections["conclusion"]
    )

    prompt = f"""
    Provide a clear and concise summary of the research paper.

    Context:
    {context}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_length=300)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)