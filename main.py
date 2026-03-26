import logging

from pathlib import Path
import os
from typing import List, Dict, Any

from config import RAGConfig, DEFAULT_CONFIG
from rag_modules import(
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_preparation = None
        self.index_construction = None
        self.retrieval_module = None
        self.generation_module = None
    
        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查环境变量是否设置
        if not os.getenv("AIHUBMIX_API_KEY"):
            raise EnvironmentError("环境变量 AIHUBMIX_API_KEY 未设置，请设置后重试。")
    
    def initialize_system(self):
        """
        初始化所有模块
        """
        print("== 初始化数据处理模块 ==")
        self.data_preparation = DataPreparationModule(data_path=self.config.data_path)
        
        print("== 初始化索引构建模块 ==")
        self.index_construction = IndexConstructionModule(
            model_path=self.config.embedding_model, 
            index_save_path=self.config.index_save_path
        )
        
        print("== 初始化生成集成模块 ==")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        print("系统初始化完成")
        
    def build_knowledge_base(self):
        """
        初始化检索优化模块
        """
        self.data_preparation.load_documents()
        chunks = self.data_preparation.chunks_documents()

        # 先尝试加载已有索引
        has_index = self.index_construction.load_index(index_path=self.config.index_save_path)
        if not has_index:
            print("未找到现有索引，正在构建新索引...")
            self.index_construction.build_vector_index(chunks=chunks)
            self.index_construction.save_index()

        vector_store = self.index_construction.vector_store

        print("== 初始化检索优化模块 ==")
        self.retrieval_module = RetrievalOptimizationModule(chunks=chunks, vectorstore=vector_store)
        
        statistics = self.data_preparation.get_statistics()
        print("=== 知识库统计信息 ===")
        print(f"文档总数: {statistics['total_documents']}")
        print(f"子文档总数: {statistics['total_chunks']}")
        print(f"平均子文档大小: {statistics['avg_chunk_size']:.2f} 字符")
    
    def answer(self, question: str, stream: bool = False) -> str:
        """"
        整理回答问题的流程，回答用户问题
        """
        if not self.retrieval_module or not self.generation_module or not self.data_preparation:
            raise ValueError("系统尚未初始化，请先调用 initialize_system() 方法进行初始化。")
        
        print(f"用户问题: {question}")
        
        # 路由判断
        router_type = self.generation_module.query_router(question)
        print(f"查询类型{router_type}")
        
        # 查询重写
        rewritten_query = question
        if router_type == "list":
            print(f"保持原来的查询{question}")
        else:
            rewritten_query = self.generation_module.query_rewriter(question)
            print(f"重写后的查询{rewritten_query}")
            
        # 获取过滤条件
        filters = self._extract_filters_from_query(question)
        print(f"提取的过滤条件: {filters}")
        
        # 检索子块
        if filters is not None:
            retrieval_chunks = self.retrieval_module.metadata_filtered_search(
                query=rewritten_query,
                filters=filters
            )
        else:
            retrieval_chunks = self.retrieval_module.hybrid_retrieve(rewritten_query)
        print(f"检索到的相关子块数量: {len(retrieval_chunks)}")
        
        # 显示检索到的子块信息
        if retrieval_chunks:
            chunk_infos = []
            for idx, chunk in enumerate(retrieval_chunks):
                dish_name = chunk.metadata.get('name', '未知菜谱')
                # 如果子块包含标题展示标题
                chunk_preview = chunk.page_content[:100].strip()
                
                if chunk_preview.startswith("#"):
                    titel_end =  chunk_preview.find("\n") if "\n" in chunk_preview else len(chunk_preview)
                    chunk_titel = chunk_preview[:titel_end].replace("#", "").strip()
                    chunk_infos.append(f"{dish_name} - {chunk_titel}")
                else:
                    chunk_infos.append(f"{dish_name} - 片段")
        else:
            return "很抱歉，未能找到相关的菜谱信息。请尝试使用不同的查询或检查您的问题是否包含特定的菜品名称、分类或难度。"
        
        print(f"找到了{len(retrieval_chunks)}个相关子块 : {",".join(chunk_infos)}")
        
        # 检索文档
        documents = self.data_preparation.get_parent_document(retrieval_chunks)
        if documents is None:
            return "很抱歉，未能找到相关的菜谱信息。请尝试使用不同的查询或检查您的问题是否包含特定的菜品名称、分类或难度。"
        
        dish_names = []
        for doc in documents:
            dish_name = doc.metadata.get('name', '未知菜谱')
            dish_names.append(dish_name)
        print(f"相关菜谱文档数量: {len(documents)}, 包含的菜谱名称: {', '.join(dish_names)}")   
        
        # 对 list 查询模式获取结果
        if router_type == "list":
            response = self.generation_module.generate_list_answer(context_docs=documents)
        # 对 detail 查询模式获取结果
        elif router_type == "detail":
            if stream:
                response = self.generation_module.generate_step_by_step_answer_stream(query=rewritten_query, context_docs=documents)
            else:
                response = self.generation_module.generate_step_by_step_answer(query=rewritten_query, context_docs=documents)
        # 对 general 模式获取结果
        elif router_type == "general":
            if stream:
                response = self.generation_module.generate_base_answer_stream(query=rewritten_query, context_docs=documents)
            else:
                response = self.generation_module.generate_basic_answer(query=rewritten_query, context_docs=documents)

        return response
    
    def _extract_filters_from_query(self, question: str) -> Dict[str, Any]:
        """
        从用户的问题中获取难度和分类过滤条件
        """
        filters = {}
        
        difficulty_list = self.data_preparation.get_supported_difficulties()        
        for difficulty in sorted(difficulty_list, key=len, reverse=True):
            if difficulty in question:
                filters["difficulty"] = difficulty
                break
            
        category_list = self.data_preparation.get_supported_categories()
        for category in category_list:
            if category in question:
                filters["category"] = category
                break
        
        return filters
    
    def run_interactive(self):
        """"
        运行一个简单的交互界面
        """
        self.initialize_system()
        self.build_knowledge_base()
        
        print("欢迎使用RAG菜谱问答系统！输入 'exit' 退出。")
        while True:
            try:
                # 接受用户问题
                question = input("\n请输入您的问题：").strip()
                if question.lower() in ["exit", "quit"]:
                    break
                
                # 接受是否使用stream
                stream_input = input("\n是否使用streaming输出，y/n, 默认不使用：").strip().lower()
                stream_input = stream_input == 'y'
                
                # 生成回答
                if stream_input:
                    for chunk in self.answer(question, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    response = self.answer(question)
                    print(f"{response}\n")
            
            except KeyboardInterrupt:
                break
            except Exception as e:  
                print(f"发生错误: {str(e)}. 请重试。")
                
        print("==感谢使用，再见！==")
    

def main():
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()
    
    except Exception as e:
        print(f"系统发生错误: {str(e)}")
        
if __name__ == "__main__":
    main()
        