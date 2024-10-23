# 环境配置

```
pip install -U pymilvus
pip install "pymilvus[model]"
```
其余环境配置参考`requirements.txt`。

# 数据库构建

构建了以下三个数据库：
- `db/huatuo_lite.db`: 基于26Mhuatuo的高质量Lite版本`FreedomIntelligence/Huatuo26M-Lite`进行构建。
- `db/medqa_all_book.db`：基于`medqa/textbooks/zh_paragraph/all_books.txt`进行构建，以4096为最大长度。
- `db/medqa_all_book_sentence.db`：基于`medqa/textbooks/zh_sentence/all_books.txt`进行构建，以4096为最大长度。

# 数据库使用

## 概述
基于`BAAI/bge-m3`，支持三种检索方式，分别为：
- `dense_search`: 基于向量检索。
- `sparse_search`: 基于文本检索。
- `hybrid_search`: 基于混合检索。


## 使用方法
在search.py中，定义了`Retriever`类，用于检索数据库。使用方法如下：

| 注意需要修改`__init__`中BGE-M3的路径，可以修改Retriever的`limit`参数来控制返回的结果数量。

```python
from search import Retriever

col_name = 'huatuo_lite' # medqa_all_book medqa_all_book_sentence
retriever = Retriever(col_name)

query_embeddings = retriever.gen_embeddings('如何治疗病毒性感冒')
dense_results = retriever.dense_search(query_embeddings["dense"][0])
sparse_results = retriever.sparse_search(query_embeddings["sparse"]._getrow(0))
hybrid_results = retriever.hybrid_search(
    query_embeddings["dense"][0],
    query_embeddings["sparse"]._getrow(0),
    sparse_weight=0.7,
    dense_weight=1.0,
)
```