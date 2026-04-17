# app.py

import streamlit as st
from rag_pipeline import *

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📄 AI-Powered Research Paper Assistant")
st.write("Upload a research paper and get summary + answers instantly 🚀")


# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    # Process PDF
    text = extract_text(uploaded_file)
    text = clean_text(text)
    sections = detect_sections(text)

    st.success("✅ PDF processed successfully!")

    # -------------------------------
    # SUMMARY SECTION
    # -------------------------------
    st.subheader("📌 Research Paper Summary")

    if st.button("Generate Section-wise Summary"):
        with st.spinner("Generating summaries..."):
            summaries = generate_section_summaries(sections)

            for sec, summ in summaries.items():
                st.subheader(f"📌 {sec.capitalize()}")
                st.write(summ)

    # -------------------------------
    # Q&A SECTION
    # -------------------------------
    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Searching and generating answer..."):

            context = retrieve_context(query, sections)

            if context.strip() == "":
                st.warning("⚠️ No relevant content found!")
            else:
                answer = generate_answer(query, context)
                st.write(answer)