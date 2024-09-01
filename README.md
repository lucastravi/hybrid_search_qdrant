#### Certifique-se de criar as chaves de acesso à API para os modelos do Gemini e Jina

--------
#### 1. Iniciando Qdrant
```
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

#### 2. Criando o banco de dados vetorial
Em um notebook dentro do projeto:
```
from hybrid_search import create_hybrid_search_db

create_hybrid_search_db()
```

#### 3. Realizando buscas no banco de dados vetorial
Em um notebook dentro do projeto:
```
from hybrid_search import make_hybrid_search

make_hybrid_search(movie_name)
```

