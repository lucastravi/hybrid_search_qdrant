import pandas as pd
from ast import literal_eval
from qdrant_client import QdrantClient, models
from sparse_search import compute_vector, load_model
import config as config

COLLECTION_NAME = "movies-hybrid-search"

model, tokenizer = load_model()

def make_movie_name_search(client, movie_name):
    """
    The function `make_movie_name_search` retrieves movie information and vectors based on a given movie
    name search query.
    :return: The function `make_movie_name_search` returns the indices and values of the query vector
    for the movie name search, as well as the dense vector associated with the search result.
    """
    result = client.scroll(collection_name=COLLECTION_NAME,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="original_title",
                                        match=models.MatchText(text=movie_name))]),
                                        with_vectors=True)
    query_text = result[0][0].payload['movie_info']
    query_vec = compute_vector(query_text, model, tokenizer)
    query_indices = query_vec.nonzero().numpy().flatten()
    query_values = query_vec.detach().numpy()[query_indices]
    result_vector = result[0][0].vector['dense-vector']
    return query_indices, query_values, result_vector

def make_hybrid_search(movie_name):
    """
    The function `make_hybrid_search` performs a hybrid search using both sparse and dense vectors to
    find similar movie names.
    :return: The function `make_hybrid_search` returns a list of tuples containing the original title
    and release year of movies that match the search query for the given `movie_name`. The search is
    performed using a hybrid approach combining sparse vector search and dense vector search techniques.
    """
    client = QdrantClient(url=config.QDRANT_URL)
    query_indices, query_values, result_vector = make_movie_name_search(client, movie_name)
    result_search = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=query_indices, values=query_values),
                    using="sparse-vector",
                    limit=20,
            ),  
                models.Prefetch(
                    query=result_vector,
                    using="dense-vector",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=10
        )
    return [(i['payload']['original_title'], 
             i['payload']['release_year']) for i in result_search.dict()['points']]

def create_hybrid_search_db():
    """
    The function `create_hybrid_search_db` creates a hybrid search database using QdrantClient with
    dense and sparse vectors from a given CSV file.
    :return: The function `create_hybrid_search_db` returns the QdrantClient object after setting up a
    hybrid search database with the specified configurations and data from the provided CSV file.
    """
    client = QdrantClient(config.QDRANT_URL, timeout=600)
    df = pd.read_csv(config.FILE_PATH)
    df['generated_text_embedding'] = df['generated_text_embedding'].apply(lambda x: literal_eval(x))
    df['sparse_vector'] = df['sparse_vector'].progress_apply(lambda x: literal_eval(x))
    vector_size = len(df['generated_text_embedding'].iloc[0])
    client.recreate_collection(
        COLLECTION_NAME,
        vectors_config={
            "dense-vector": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse-vector": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
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
    for index, row in df.iterrows():
        indices = row.sparse_vector.keys()
        values = row.sparse_vector.values()
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=index,
                    vector={
                        "dense-vector": row["generated_text_embedding"],
                        "sparse-vector": models.SparseVector(
                            indices=indices, values=values)
                    },
                    payload={
                        "original_title": row["original_title"],
                        "original_language": row["original_language"],
                        "release_year": row["release_year"],
                        "movie_info": row["generated_text"]
                    }
                )
            ]
        )
    return client

