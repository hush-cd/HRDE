## Preloaded embedding model

from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    'moka-ai/m3e-large', 
    device='cuda:0'
)
model.save('./m3e-large')
