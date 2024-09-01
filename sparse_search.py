from qdrant_client import QdrantClient, models
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from ast import literal_eval
import pandas as pd
from tqdm import tqdm
import config as config

COLLECTION_NAME = "sparse_vectors"

def make_sparse_vector_query(movie_name):
    """
    The function `make_sparse_vector_query` uses QdrantClient to search for similar movie titles based
    on a given movie name.
    :return: The function `make_sparse_vector_query(movie_name)` returns a list of original movie titles
    that are similar to the input `movie_name` based on a sparse vector query using QdrantClient.
    """
    client = QdrantClient(url=config.QDRANT_URL)
    result = client.scroll(collection_name=COLLECTION_NAME,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="original_title",
                                        match=models.MatchText(text=movie_name))]))
    query_text = result[0][0].payload['movie_info']
    query_vec = compute_vector(query_text)
    query_indices = query_vec.nonzero().numpy().flatten()
    query_values = query_vec.detach().numpy()[query_indices]
    results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=models.NamedSparseVector(
                        name="text",
                        vector=models.SparseVector(
                            indices=query_indices,
                            values=query_values,
                        ),
                    ),
                    with_vectors=False, 
                    limit=10)
    return [i.payload['original_title'] for i in results]


def create_sparse_vectors(df):
    """
    The function `create_sparse_vectors` processes a DataFrame by computing sparse vectors for each
    row's 'generated_text' column and storing the result in a new 'sparse_vector' column.
    :return: The function `create_sparse_vectors` is returning the input DataFrame `df` with an
    additional column 'sparse_vector' that contains the sparse vectors computed from the
    'generated_text' column.
    """
    model, tokenizer = load_model()
    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        vec = compute_vector(row['generated_text'], model, tokenizer)
        indices = vec.nonzero().numpy().flatten()
        values = vec.detach().numpy()[indices]
        df.loc[ix, 'sparse_vector'] = str({k:v for k,v in zip(indices, values)})
    return df


def create_qdrant_sparse_db():
    """
    The function `create_qdrant_sparse_db` sets up a Qdrant client, reads data from a CSV file,
    processes sparse vectors, and inserts them into a Qdrant collection along with additional metadata.
    :return: The function `create_qdrant_sparse_db()` returns the Qdrant client object after setting up
    the client, reading data from a CSV file, processing the sparse vectors, creating a collection in
    Qdrant with specified configurations, creating a payload index, and upserting data points into the
    collection.
    """
    tqdm.pandas()
    client = QdrantClient(url=config.QDRANT_URL)
    df = pd.read_csv(config.FILE_PATH)
    df['sparse_vector'] = df['sparse_vector'].progress_apply(lambda x: literal_eval(x))
    client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    })
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="original_title",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=100,
            lowercase=True,
    ))
    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        indices = row.sparse_vector.keys()
        values = row.sparse_vector.values()
        client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=ix,
                payload={
                    "original_title": row["original_title"],
                    "original_language": row["original_language"],
                    "release_year": row["release_year"],
                    "movie_info": row["generated_text"]
                },
                vector={
                    "text": models.SparseVector(
                        indices=indices, values=values
                    )
                },
            )
        ])
    return client


def load_model():
    """
    The `load_model` function loads a pre-trained masked language model and tokenizer from the Hugging
    Face model hub.
    :return: The function `load_model()` returns a pre-trained masked language model and its
    corresponding tokenizer.
    """
    model_id = 'naver/splade-cocondenser-ensembledistil'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    return model, tokenizer


def compute_vector(text, model, tokenizer):
    """
    The function `compute_vector` takes a text input, processes it using a loaded model and tokenizer,
    and computes a vector representation based on the model's output.
    :return: The function `compute_vector(text)` returns a vector representation of the input text after
    processing it through a loaded model and tokenizer. The vector is computed based on the maximum
    value of the weighted logarithm of the model's output logits, considering the attention mask of the
    tokens in the text.
    """
    tokens = tokenizer(text, return_tensors="pt", 
                       padding=True, truncation=True, 
                       add_special_tokens = True)
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()
    return vec

def extract_and_map_sparse_vector(vector, tokenizer):
    """
    The function `extract_and_map_sparse_vector` takes a sparse vector and a tokenizer, extracts
    non-zero elements, maps them to tokens, and returns a sorted dictionary of tokens with their
    corresponding weights.
    :return: The function `extract_and_map_sparse_vector` takes a sparse vector and a tokenizer as
    input. It extracts the non-zero elements and their corresponding weights from the vector, maps the
    indices to tokens using the tokenizer, and returns a dictionary where tokens are keys and weights
    are values. The dictionary is sorted by weight in descending order before being returned.
    """
    cols = vector.nonzero().squeeze().cpu().tolist()
    weights = vector[cols].cpu().tolist()
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    token_weight_dict = {idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)}
    sorted_token_weight_dict = \
        {k: v for k, v in sorted(token_weight_dict.items(), 
                                 key=lambda item: item[1], reverse=True)}
    return sorted_token_weight_dict