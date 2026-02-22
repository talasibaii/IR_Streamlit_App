import re
import streamlit as st
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# --- stopwords (no nltk needed) ---
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by",
    "with","from","as","is","are","was","were","be","been","being","it","this","that","these",
    "those","i","you","he","she","we","they","them","his","her","our","their","my","your"
}

def tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

# --- load docs + embeddings + model ---
embeddings = np.load("embeddings.npy")
model = Word2Vec.load("word2vec.model")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

def get_query_embedding(query: str):
    tokens = tokenize(query)
    vecs = [model.wv[w] for w in tokens if w in model.wv]

    # same idea as your Lab2: average word vectors :contentReference[oaicite:3]{index=3}
    if len(vecs) == 0:
        return None, tokens

    return np.mean(vecs, axis=0).astype(np.float32), tokens

def retrieve_top_k(query_embedding, embeddings, k=10):
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(documents[i], float(sims[i])) for i in top_idx]

# --- UI ---
st.title("Information Retrieval using Word2Vec Embeddings")

query = st.text_input("Enter your query:")
k = st.slider("Top K results", min_value=1, max_value=10, value=5)

if st.button("Search"):
    query_vec, tokens = get_query_embedding(query)


    if query_vec is None:
        st.error("None of your query words exist in the Word2Vec vocabulary. Try different words.")
    else:
        results = retrieve_top_k(query_vec, embeddings, k=k)
        st.write(f"### Top {k} Relevant Documents:")
        for doc, score in results:
            st.write(f"- **{doc}** (Score: {score:.4f})")
