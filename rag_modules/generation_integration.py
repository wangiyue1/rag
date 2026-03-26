import logging
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .data_preparation import DataPreparationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    def __init__(self, model_name: str = "glm-4.7-flash-free", temperature: float = 0.1, max_tokens:int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._setup_llm()
        
    def _setup_llm(self):
        api_key = os.getenv("AIHUBMIX_API_KEY")
        if not api_key:
            raise ValueError("API key 不存在")
        self.llm = ChatOpenAI(
            model= self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url="https://aihubmix.com/v1"
        )
        
        logger.info("LLM 已成功初始化，使用模型: %s", self.model_name)
        
    def generate_basic_answer(self, query: str, context_docs: List[Document])->str:
        """
        生成基本回答
        """
        prompt = ChatPromptTemplate.from_template("""
        你是一名专业的烹饪助手。请根据以下食谱信息回答用户的问题：
        用户问题：{question}
        
        相关食谱信息：{context}
        请提供真实可靠的信息，若信息不足请诚实说明。
        """)
        context = self._build_context(context_docs)
        
        chain = (
            prompt 
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke({"question": query, "context": context})
        return response

    def generate_base_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
        你是一名专业的烹饪助手。请根据以下食谱信息回答用户的问题：
        用户问题：{question}
        
        相关食谱信息：{context}
        请提供真实可靠的信息，若信息不足请诚实说明。
        """)
        context = self._build_context(context_docs)
        
        chain = (
            prompt 
            | self.llm
            | StrOutputParser()
        )
        
        for response in chain.stream({"question": query, "context": context}):
            yield response
        
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        
        prompt = ChatPromptTemplate.from_template("""
        你是一名专业的烹饪助手。请根据以下食谱信息帮助用户生成详细的烹饪步骤：
        用户问题：{question}
        
        相关食谱信息：{context}
        
        请灵活组织回答，建议包含以下步骤：
        - 菜品介绍
            【简要介绍菜品名称和制作难度】
        - 食材准备
            【列出所需食材和用量】
        - 烹饪步骤
            【详细描述每一步的操作方法和操作时间】
        - 小贴士
            【提供一些烹饪技巧或者注意事项， 优先使用从提供的食谱相关信息获取，如果没有可进行适当总结，或者可省略】
        
        注意：
        1.可以灵活调整结构，不要重复或者强行填充
        2. 小贴士没有合适的可以省略
        3. 省略的内容不需要告诉用户，查找食谱的过程也不需要告诉用户
        """)
        
        context = self._build_context(context_docs)
        
        chain = (
            prompt 
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke({"question": query, "context": context})
        return response

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
            
            prompt = ChatPromptTemplate.from_template("""
            你是一名专业的烹饪助手。请根据以下食谱信息帮助用户生成详细的烹饪步骤：
            用户问题：{question}
            
            相关食谱信息：{context}
            
            请灵活组织回答，建议包含以下步骤：
            - 菜品介绍
                【简要介绍菜品名称和制作难度】
            - 食材准备
                【列出所需食材和用量】
            - 烹饪步骤
                【详细描述每一步的操作方法和操作时间】
            - 小贴士
                【提供一些烹饪技巧或者注意事项， 优先使用从提供的食谱相关信息获取，如果没有可进行适当总结，或者可省略】
            
            注意：
            1.可以灵活调整结构，不要重复或者强行填充
            2. 小贴士没有合适的可以省略
            """)
            
            context = self._build_context(context_docs)
            
            chain = (
                prompt 
                | self.llm
                | StrOutputParser()
            )
            
            for chunk in chain.stream({"question": query, "context": context}):
                yield chunk

    def query_rewriter(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
            你是一名专业的用户输入改写助手。请你判断用户输入的语句是否需要改写，以便准确检索到相关的食谱信息。
            查询语句：{query}
            
            #判断原则#
            1. **具体查询**：不需要改写
                - 包含具体的菜名，如“红烧肉的做法”， “宫保鸡丁需要什么食材
                - 询问烹饪技巧，如“如何炒出香喷喷的菜”， “如何判断菜熟了没”
            2. **模糊查询**：需要改写
                -过于宽泛、不具体的：如“怎么做川菜”， “有什么好吃的”， “推荐个菜”
            
            #改写原则#
            1. 保持原意
            2. 添加具体信息
            3. 推荐简单的
            4. 保持简介
            
            示例：
            - “怎么做川菜” 输出：“有什么经典简单的川菜菜谱”
            - “有什么好吃的” 输出：“有什么好吃的简单家常菜”
            - “推荐个菜” 输出：“推荐个简单的家常菜”
            - “红烧肉的做法” 输出：“红烧肉的做法”
            - “如何炒出香喷喷的菜” 输出：“如何炒出香喷喷的家常菜”
            
            仅输出改写结果，不要输出判断结果或者其他无关内容
            """,
            input_variables=["query"]
        )
        
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        rewritten_query = chain.invoke({"query": query})
        
        if rewritten_query == query:
            logger.info("查询不需要改写")
        else:
            logger.info("查询已改写为: %s", rewritten_query)
        return rewritten_query

    def query_router(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
            请将用户的问题分为以下三类：
            1. "list" : 用户想获取菜品列表或者推荐，只需要菜名
                例如：“推荐个素菜”， “怎么做川菜”
            2. "detail" : 用户想获取具体的步骤或者详细信息
                例如： “红烧肉的做法”， “宫保鸡丁需要什么食材”
            3. "general": 其他问题
                例如： “如何判断菜熟了没”， “什么是川菜”
                
            用户问题：{query}
            只需要返回以下类型： "list", "detail", "general"
            """,
            input_variables=["query"]
        )
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        category = chain.invoke({"query": query})
        
        if category not in ['list', 'detail', 'general']:
            logger.warning(f"返回了未知类型：{category}")
            category = "general"
        return category
    
    def _build_context(self, context_docs: List[Document], max_length: int = 2000)->str:
        """"
        构建上下文信息：
        将Documents进行整理成一段文字：
            - 元数据：食谱序号 | 食谱名称 | 食谱类型 | 难度
            - 内容
        """
        if not context_docs:
            return "暂无相关信息"
        
        context_parts = []
        current_length = 0
        for idx, doc in enumerate(context_docs):
            meta_info = f"食谱{idx}"
            if "name" in doc.metadata:
                meta_info += f" | 食谱名称：{doc.metadata.get('name')}"
            if "category" in doc.metadata:
                meta_info += f" | 食谱类型：{doc.metadata.get('category')}"
            if "difficulty" in doc.metadata:
                meta_info += f" | 难度：{doc.metadata.get('difficulty')}"
            
            doc_info = f"{meta_info}\n{doc.page_content}\n"
            
            if current_length + len(doc_info) > max_length:
                break
            
            current_length += len(doc_info)
            context_parts.append(doc_info)
        
        return "\n" + "=" * 50 + "\n".join(context_parts)
    
    def generate_list_answer(self, context_docs: List[Document]) -> str:
        """
        生成列表类回答：
        """
        if context_docs is None:
            return "没有找到相关信息"
        
        meal_names = []
        for idx, doc in enumerate(context_docs):
            name = doc.metadata.get("name", "未知菜谱")
            if name not in meal_names:
                meal_names.append(name)
        
        if len(meal_names) == 0:
            return "没有找到相关信息"
        elif len(meal_names) <= 3:
            return f"根据您的查询， 为您推荐以下菜品： \n"  + "\n".join([f"{idx + 1}.{name}" for idx, name in enumerate(meal_names)])
        else:
            return f"根据您的查询，为您推荐以下菜品: \n" + "\n".join([f"{idx + 1}.{name}" for idx, name in enumerate(meal_names[:3])]) + f"\n等{len(meal_names) - 3}道菜品，您可以尝试更具体的查询来获取更多相关菜品信息。"
        
if __name__ == "__main__":
    data_preparation = DataPreparationModule() 
    documents = data_preparation.load_documents()
    chunks = data_preparation.chunks_documents()
    
    index_construction = IndexConstructionModule()
    if not index_construction.load_index("./index"):
        index_construction.build_vector_index(chunks=chunks)
        index_construction.save_index()
    
    retrieval_module = RetrievalOptimizationModule(chunks=chunks, vectorstore=index_construction.vector_store)
    context = retrieval_module.hybrid_retrieve("牛肉")
    
    generation_module = GenerationIntegrationModule()
    
    # 测试生成功能
    # result = generation_module.generate_step_by_step_answer("酱牛肉怎么做", context)
    # 流式输出示例
    print("生成的步骤如下：")
    for chunk in generation_module.generate_step_by_step_answer_stream("酱牛肉怎么做", context):
        print(chunk, end="", flush=True)
    print("\n生成完成！")
    # result = generation_module.generate_list_answer(context)
    
    # result = generation_module.generate_basic_answer("怎么判断油温", context)
    # print(result)
    
    # 测试改写功能
    # result = generation_module.query_rewriter("推荐个菜")
    
    # print(result)
    
    # 测试路由功能
    # category = generation_module.query_router("怎么判断油温")
    
    # category = generation_module.query_router("红烧肉的做法")
    
    # category = generation_module.query_router("推荐几个川菜")

