# # rag_pipeline.py

# import fitz
# import re
# import numpy as np
# import faiss

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# # -------------------------------
# # LOAD MODELS
# # -------------------------------
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# # -------------------------------
# # PDF EXTRACTION
# # -------------------------------
# def extract_text(uploaded_file):
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text


# # -------------------------------
# # CLEAN TEXT (IMPORTANT FIX)
# # -------------------------------
# def clean_text(text):

#     # fix broken words
#     text = text.replace("-\n", "")

#     # ❌ DO NOT REMOVE \n (needed for section detection)

#     # remove emails
#     text = re.sub(r'\S+@\S+', '', text)

#     # remove "given name surname" patterns
#     text = re.sub(r'given name surname\d*', '', text.lower())

#     return text


# # -------------------------------
# # SPLIT SENTENCES
# # -------------------------------
# def split_sentences(text):

#     sentences = re.split(r'(?<=[.!?])\s+', text)

#     cleaned = []

#     for s in sentences:
#         s = s.strip()

#         if len(s) < 80:
#             continue
#         if "et al" in s.lower():
#             continue
#         if "[" in s and "]" in s:
#             continue
#         if s and s[0].isdigit():
#             continue

#         cleaned.append(s)

#     return cleaned


# # -------------------------------
# # SECTION DETECTION (FIXED)
# # -------------------------------
# def detect_sections(text):

#     sections = {
#         "abstract": "",
#         "introduction": "",
#         "method": "",
#         "results": "",
#         "conclusion": ""
#     }

#     current = None

#     lines = text.split("\n")

#     for line in lines:
#         l = line.strip().lower()

#         if l.startswith("abstract"):
#             current = "abstract"
#             continue
#         elif l.startswith("introduction"):
#             current = "introduction"
#             continue
#         elif "method" in l or "proposed" in l:
#             current = "method"
#             continue
#         elif "result" in l or "experiment" in l:
#             current = "results"
#             continue
#         elif "conclusion" in l or "discussion" in l:
#             current = "conclusion"
#             continue

#         if current:
#             sections[current] += " " + line

#     return sections


# # -------------------------------
# # QUERY → SECTION MAPPING
# # -------------------------------
# def get_section_from_query(query):

#     q = query.lower()

#     if "method" in q:
#         return "method"
#     elif "result" in q:
#         return "results"
#     elif "conclusion" in q:
#         return "conclusion"
#     elif "abstract" in q:
#         return "abstract"
#     else:
#         return "introduction"


# # -------------------------------
# # RAG RETRIEVAL
# # -------------------------------
# def retrieve_context(query, sections):

#     target_section = get_section_from_query(query)
#     section_text = sections[target_section]

#     sentences = split_sentences(section_text)

#     if len(sentences) == 0:
#         return ""

#     embeddings = embed_model.encode(sentences)

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))

#     query_vec = embed_model.encode([query])

#     D, I = index.search(np.array(query_vec), k=min(6, len(sentences)))

#     retrieved = [sentences[i] for i in I[0]]

#     return " ".join(retrieved[:5])


# # -------------------------------
# # ANSWER GENERATION (IMPROVED)
# # -------------------------------
# def generate_answer(query, context):

#     prompt = f"""
#     You are an expert AI assistant for research papers.

#     Answer clearly and in detail using ONLY the context.
#     If the answer is not present, say "Not found in document".

#     Question: {query}

#     Context:
#     {context}
#     """

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

#     outputs = llm_model.generate(
#         **inputs,
#         max_length=300,
#         do_sample=False
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# # -------------------------------
# # SECTION-WISE SUMMARY (UPGRADED)
# # -------------------------------
# def generate_section_summaries(sections):

#     summaries = {}

#     for section, content in sections.items():

#         if len(content.strip()) < 50:
#             continue

#         # remove author noise
#         content = re.sub(r'author.*?;', '', content, flags=re.IGNORECASE)

#         prompt = f"""
#         You are an expert AI assistant.

#         Ignore author names, emails, and template text.

#         Summarize the following section of a research paper in detail.
#         Provide clear explanation and key points.

#         Section: {section}

#         Content:
#         {content[:3000]}
#         """

#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

#         outputs = llm_model.generate(
#             **inputs,
#             max_length=400,
#             min_length=150,
#             do_sample=False
#         )

#         summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         summaries[section] = summary

#     return summaries

# rag_pipeline.py

import fitz
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# LOAD MODELS
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# -------------------------------
# PDF EXTRACTION
# -------------------------------
def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = text.replace("-\n", "")

    # remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # remove template noise
    text = re.sub(r'given name surname\d*', '', text.lower())

    return text


# -------------------------------
# SPLIT SENTENCES
# -------------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if len(s) < 50:
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

    current = None

    for line in text.split("\n"):
        l = line.strip().lower()

        if l.startswith("abstract"):
            current = "abstract"
            continue
        elif l.startswith("introduction"):
            current = "introduction"
            continue
        elif "method" in l or "approach" in l:
            current = "method"
            continue
        elif "result" in l or "experiment" in l:
            current = "results"
            continue
        elif "conclusion" in l or "discussion" in l:
            current = "conclusion"
            continue

        if current:
            sections[current] += " " + line

    return sections


# -------------------------------
# SENTENCE RANKING (NEW ⭐)
# -------------------------------
def rank_sentences(sentences, top_k=10):

    if len(sentences) <= top_k:
        return sentences

    embeddings = embed_model.encode(sentences)

    # similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    scores = sim_matrix.sum(axis=1)

    ranked_indices = np.argsort(scores)[::-1]

    top_sentences = [sentences[i] for i in ranked_indices[:top_k]]

    return top_sentences


# -------------------------------
# QUERY → SECTION
# -------------------------------
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

    section = get_section_from_query(query)
    text = sections[section]

    sentences = split_sentences(text)

    if not sentences:
        return ""

    embeddings = embed_model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query_vec = embed_model.encode([query])

    _, I = index.search(np.array(query_vec), k=min(6, len(sentences)))

    retrieved = [sentences[i] for i in I[0]]

    return " ".join(retrieved[:5])


# -------------------------------
# ANSWER GENERATION
# -------------------------------
def generate_answer(query, context):

    prompt = f"""
    You are an expert AI assistant for research papers.

    Answer clearly using ONLY the context.
    Avoid repetition.

    Question: {query}

    Context:
    {context}
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(
        **inputs,
        max_length=300,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# HYBRID SECTION SUMMARY ⭐
# -------------------------------
def generate_section_summaries(sections):

    summaries = {}

    for section, content in sections.items():

        if len(content.strip()) < 50:
            continue

        # clean noise
        content = re.sub(r'author.*?;', '', content, flags=re.IGNORECASE)

        # STEP 1: sentence splitting
        sentences = split_sentences(content)

        # STEP 2: rank important sentences ⭐
        important = rank_sentences(sentences, top_k=12)

        filtered_text = " ".join(important)

        # STEP 3: LLM summarization
        prompt = f"""
        Summarize the following research section clearly.

        Avoid repetition.
        Give structured key points.

        Section: {section}

        Content:
        {filtered_text}
        """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = llm_model.generate(
            **inputs,
            max_length=300,
            min_length=120,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        summaries[section] = summary

    return summaries