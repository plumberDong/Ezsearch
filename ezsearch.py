import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pandas as pd
import os
import openpyxl
import datetime
import api_keys # the file containing the API key for OpenAI

#--------------
# 搜索参数
#--------------
query_texts = '算法社会' # 思路描述
n_results = 15 # number of results to return

#---------------
# 主程序部分
#---------------

def main():
   # creating a permanent client
    client = chromadb.PersistentClient(path = 'chromadb')

    collection = client.get_collection(name="sociology_papers", 
                                        embedding_function=OpenAIEmbeddingFunction(
                                            api_key = api_keys.api_key[0],
                                            api_base= api_keys.api_base[0],
                                            model_name = "text-embedding-ada-002"))

    # return the top results for the query text
    result = collection.query(
        query_texts = [query_texts],
        n_results = n_results
    )
    df = pd.DataFrame(result['metadatas'][0])
    
    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # 构造文件名
    file_path = f"./output/{current_time}.xlsx"
    
    # 检查文件夹位置
    if not os.path.exists('output'):
        os.makedirs('output')
        
    # 存储文件
    df.to_excel(file_path, index=False)
    
if __name__ == "__main__":
    main()


