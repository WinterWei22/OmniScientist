import os
import pickle
import numpy as np
from openai import OpenAI
from ...config import template_embedding_api_key, template_embedding_api_base_url

current_dir = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(current_dir, "templates.pkl")
with open(pickle_dir, "rb") as f:
    templates = pickle.load(f)
    f.close()

client = OpenAI(
    api_key=template_embedding_api_key,
    base_url=template_embedding_api_base_url,
)


def get_embedding(text):
    resp = client.embeddings.create(
        model="ep-20250514195010-6l4kl", input=[text], encoding_format="float"
    )
    embedding = np.array(resp.data[0].embedding)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def retrieve_small_template(query):
    query_embedding = get_embedding(query)
    score = np.dot(query_embedding, templates["small"]["embeddings"].T)
    index = np.argmax(score)
    return templates["small"]["value_list"][index]


def retrieve_large_template(query):
    query_embedding = get_embedding(query)
    score = np.dot(query_embedding, templates["large"]["embeddings"].T)
    index = np.argmax(score)
    return templates["large"]["value_list"][index]
