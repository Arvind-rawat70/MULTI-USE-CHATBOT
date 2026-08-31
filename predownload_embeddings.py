from sentence_transformers import SentenceTransformer

print("Downloading sentence-transformers/all-MiniLM-L6-v2 ...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Done. Model cached locally.")

vec = model.encode("hello world")
print("Embedding vector length:", len(vec))
