from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    client,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class Retriever:
    def __init__(self, col_name , limit=10):
        self.ef = BGEM3EmbeddingFunction(
            model_name='/data/model/BAAI/bge-m3', # Specify the model name
            device='cuda:8', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )
        connections.connect(uri=f"db/{col_name}.db")
        self.col = Collection(col_name)
        self.limit = limit
        
    def gen_embeddings(self, text):
        return self.ef([text])
    
    def dense_search(self, query_dense_embedding):
        search_params = {"metric_type": "IP", "params": {}}
        res = self.col.search(
            [query_dense_embedding],
            anns_field="dense_vector",
            limit=self.limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def sparse_search(self, query_sparse_embedding):
        search_params = {
            "metric_type": "IP",
            "params": {},
        }
        res = self.col.search(
            [query_sparse_embedding],
            anns_field="sparse_vector",
            limit=self.limit,
            output_fields=["text"],
            param=search_params,
        )[0]
        return [hit.get("text") for hit in res]

    def hybrid_search(
        self,
        query_dense_embedding,
        query_sparse_embedding,
        sparse_weight=1.0,
        dense_weight=1.0,
    ):
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [query_dense_embedding], "dense_vector", dense_search_params, limit=self.limit
        )
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=self.limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search(
            [sparse_req, dense_req], rerank=rerank, limit=self.limit, output_fields=["text"]
        )[0]
        return [hit.get("text") for hit in res]
    
# Example usage
if __name__ == "__main__":
    retriever = Retriever("huatuo_lite", limit=3)
    query_embeddings = retriever.gen_embeddings('如何治疗病毒性感冒')
    dense_results = retriever.dense_search(query_embeddings["dense"][0])
    sparse_results = retriever.sparse_search(query_embeddings["sparse"]._getrow(0))
    hybrid_results = retriever.hybrid_search(
        query_embeddings["dense"][0],
        query_embeddings["sparse"]._getrow(0),
        sparse_weight=0.7,
        dense_weight=1.0,
    )
    print("Dense search results:", dense_results)
    print("Sparse search results:", sparse_results)
    print("Hybrid search results:", hybrid_results)