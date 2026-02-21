import re
import numpy as np
from gensim.models import Word2Vec

# Small stopwords list (so we don't need NLTK downloads)
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by",
    "with","from","as","is","are","was","were","be","been","being","it","this","that","these",
    "those","i","you","he","she","we","they","them","his","her","our","their","my","your"
}

def tokenize(text: str):
    # keep only letters/numbers, lowercase
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

# 1) Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# 2) Tokenize
sentences = [tokenize(doc) for doc in documents]

# 3) Train Word2Vec on your own documents
# NOTE: min_count=1 because you have a small dataset
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    sg=1  # skip-gram
)

# 4) Build document embeddings by averaging word vectors (like your Lab2 approach)
def doc_embedding(tokens):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

embeddings = np.vstack([doc_embedding(toks) for toks in sentences]).astype(np.float32)

# 5) Save files for Streamlit
np.save("embeddings.npy", embeddings)
model.save("word2vec.model")

print("✅ Saved embeddings.npy and word2vec.model")
print("Embeddings shape:", embeddings.shape)
