import qdrant_client
import requests
from qdrant_client.models import Distance, VectorParams, Batch
from ast import literal_eval

# Provide Jina API key and choose one of the available models.
# You can get a free trial key here: https://jina.ai/embeddings/

def create_text_embedding(documents):
    JINA_API_KEY = "jina_xxxxxxxxxxx"
    MODEL = "jina-embeddings-v2-base-en"  # or "jina-embeddings-v2-base-en"

    # Get embeddings from the API
    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }

    data = {
        "input": documents,
        "model": MODEL,
    }
    response = requests.post(url, headers=headers, json=data)
    embeddings = [d["embedding"] for d in response.json()["data"]]
    return embeddings

def preprocess_data(df):
    df = get_col_names(df, 'genres')
    df = get_col_names(df, 'keywords')
    # df['movie_info'] = df.apply(lambda x: x['overview'], axis=1)
    return df


def fasdfasdasda():
    return None

def get_col_names(df, col):
    df[col] = df[col].apply(lambda x: literal_eval(x))
    df[f'{col}_name'] = df[col].apply(lambda x: ', '.join([i['name'] for i in x]))
    return df
