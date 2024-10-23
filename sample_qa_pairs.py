import random
from pymilvus import connections, Collection

def connect_to_database(db_path):
    connections.connect(uri=db_path)
    return Collection("huatuo_lite")

def get_collection_schema(collection):
    schema = collection.schema
    print("Collection Schema:", schema)
    return schema

def retrieve_all_data(collection):
    offset = 0
    max_batch_size = 16384  # 最大批次大小
    all_data = []
    total_count = collection.num_entities 
    
    while offset < total_count:
        limit = min(max_batch_size - offset, total_count - offset)
        if limit <= 0:
            break
        # print(f"Querying from offset: {offset}, limit: {limit}")  # 打印查询偏移量和批次大小
        res = collection.query(expr="text != ''", offset=offset, limit=limit, output_fields=["text"])
        all_data.extend([item["text"] for item in res])
        offset += limit
    
    return all_data


def save_to_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(f"{item}\n")

if __name__ == "__main__":
    db_path = "huatuo_lite.db"
    collection = connect_to_database(db_path)
    schema_info = get_collection_schema(collection)
    all_data = retrieve_all_data(collection)

    if len(all_data) >= 500:
        sampled_data = random.sample(all_data, 500)
    else:
        print("数据量不足，无法抽样 500 条。")
        sampled_data = all_data

    save_to_file(sampled_data, "sampled_qa_pairs.txt")
    print("已成功抽样并保存 500 条问答对到 sampled_qa_pairs.txt")
