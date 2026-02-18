import numpy as np

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

embedding_dim = 512
num_documents = len(documents)

# Generate random embeddings
document_embeddings = np.random.rand(num_documents, embedding_dim).astype(np.float32)

# Save embeddings
np.save("embeddings.npy", document_embeddings)

print("Embeddings file created successfully!")
