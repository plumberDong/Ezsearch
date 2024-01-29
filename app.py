from shiny import ui, render, App, reactive
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pandas as pd
import utils.api_keys as api_keys # the file containing the API key for OpenAI

# 定义文献源
choices = {"社会学研究": "社会学研究", "社会": "社会"} #

# 连接向量数据库
client = chromadb.PersistentClient(path = 'chromadb')

collection = client.get_collection(name="sociology_papers", 
                                   embedding_function=OpenAIEmbeddingFunction(
                                    api_key = api_keys.api_key[0],
                                    api_base= api_keys.api_base[0],
                                    model_name = "text-embedding-ada-002"))

# 前端部分
app_ui = ui.page_fluid(
    ui.h1('EZsearch_V1.0'),
    ui.markdown(
        '''
        *基于语义的社科文献检索工具*
        
        **使用说明**：
        1. 文本框输入你感兴趣话题的一段`简单陈述`（不多于300字）。例如："中国式现代化的一个重要阻碍是基层治理能力不足"。
        2. 点击`搜索`按钮。程序会自动搜索话题相关文献并排序。请耐心等候！  
        项目地址：[Github](https://github.com/plumberDong/Ezsearch)
        '''
    ),
    
    # input区
    ui.panel_well(
        ui.row(
            # text input
            ui.column(5, 
                ui.input_text_area("text", "输入你的陈述：", placeholder="Enter text", rows=8),
                ui.input_action_button("gobutton", "搜索", class_="btn-primary")
            ),
            ui.column(4, 
                ui.input_slider('num_of_papers', '返回文献数量', value = 10, max=30, min = 5),
                ui.input_checkbox_group("source", "文献源", choices, selected=["社会学研究", "社会"])
            )
        )
    ),
        
    # 分割线
    ui.hr(),
    
    # 输出区
    ui.output_table('paper_table')
)

# 后端部分
def server(input, output, session):

    @reactive.calc()
    @reactive.event(input.gobutton) # 点击gobutton后再生成结果
    # 进行文献检索
    def search_papers():
        result = collection.query(
            query_texts = [input.text()], # 简单陈述
            n_results = input.num_of_papers(), # 返回的文献数量
            # 筛选条件
            where={
                    "Source": {
                        "$in": list(input.source())
                    }
                    }
            )
        df = pd.DataFrame(result['metadatas'][0])
        
        # selected columns
        df = df[['Title', 'Author', 'Source', 'Year', 'Volume', 'Keyword']]
        
        return df
    
    
    # 输出表格
    @output
    @render.table
    def paper_table():
        return search_papers()


# This is a shiny.App object. It must be named `app`.
app = App(app_ui, server)

