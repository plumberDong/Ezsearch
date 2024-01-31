import chromadb
import api_keys as api_keys
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import time

# the file path of the xls file containing the papers to be added
dat_path = 'datasets/sh_2023_06.csv'


# creating a permanent client
client = chromadb.PersistentClient(path = 'chromadb')

# connecting to the  collection with OpenAI embedding function
collection = client.get_collection(name="sociology_papers", 
                                      embedding_function=OpenAIEmbeddingFunction(
                                          api_key = api_keys.api_key[0],
                                          api_base= api_keys.api_base[0],
                                          model_name = "text-embedding-ada-002"))

# reading the xls file into a pandas dataframe
df = pd.read_csv(dat_path, encoding = 'gbk')

# converting the dataframe to a list of dictionaries
df_to_dict = df.to_dict(orient='records')

# adding the papers to the collection
ids, documents, metadatas = [], [], []
for i, doc in enumerate(df_to_dict):
    ids.append(doc['id'])
    
    text = f"""题目:{doc['Title']}
    摘要:{doc['Summary']}
    关键词:{doc['Keyword']}"""
    
    documents.append(text)
    doc['Summary'] = '' # 向量数据库不保存summary，以节省空间
    metadatas.append(doc)
    
# 添加记录
# 设置批次大小和暂停时间
batch_size = 50  # 每批次处理的记录数
pause_time = .2  # 每个批次之间的暂停时间（秒）

# 计算需要的批次数量
num_batches = (len(ids) + batch_size - 1) // batch_size

for i in range(num_batches):
    # 计算当前批次的起始和结束索引
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    # 获取当前批次的数据
    batch_ids = ids[start_idx:end_idx]
    batch_documents = documents[start_idx:end_idx]
    batch_metadatas = metadatas[start_idx:end_idx]

    # 添加当前批次的记录
    try:
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        print(f"Batch {i+1}/{num_batches} successfully added.")
    except Exception as e:
        print(f"An error occurred: {e}")
        break  # 停止处理，因为遇到错误

    # 如果这不是最后一批，暂停一会儿
    if i < num_batches - 1:
        print(f"Pausing for {pause_time} seconds before the next batch...")
        time.sleep(pause_time)