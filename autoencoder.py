from sentence_transformers import SentenceTransformer
import numpy as np
from parameters import *


def embed(texts, save_dir=None, read_dir=None):
    if read_dir:
        return np.load(read_dir)
    roberta = SentenceTransformer('stsb-roberta-large')
    embedding = roberta.encode(texts, convert_to_tensor=True).cpu().numpy()
    if save_dir:
        np.save(save_dir, embedding)
    return embedding
