import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from .index_construction import IndexConstructionModule
from .data_preparation import DataPreparationModule

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    def __init__(self, chunks: List[Document], vectorstore: FAISS):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self._setup_retriever()
    
    def _setup_retriever(self):
        logger.info("设置检索器...")
        # 向量检索器
        self.embedding_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3} 
        )
        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )
        logger.info("检索器设置完成")
        
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k :int = 5)->List[Document]:
        """
        根据元数据对检索结果进行过滤
            
        Args:
            filters: 例如{"category": "["水产", "荤菜"]", "difficulty": "简单"}    
        """
        reranked_docs = self.hybrid_retrieve(query)
        
        filtered_docs = []
        for doc in reranked_docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
                    
        return filtered_docs

    def hybrid_retrieve(self, query: str, top_k: int = 3)->List[Document]:
        """获取检索结果"""
        vector_docs = self.embedding_retriever.invoke(query)
        # bm25_docs = self.bm25_retriever.invoke(query)
        
        # reranked_docs = self.__rrf_rank(vector_docs, bm25_docs)
        # return reranked_docs[:top_k]
        return vector_docs[:top_k]
        
    def __rrf_rank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
            从向量检索和BM25检索的结果中，使用RRF算法进行融合排序
        """
        docs_score = {}
        docs_object = {}
        
        # 计算向量检索结果的RRF分数
        for idx, doc in enumerate(vector_docs):
            # 使用文档内容的hash值作为唯一标识
            doc_id = hash(doc.page_content)
            docs_object[doc_id] = doc
            rrf = 1 / (k + idx + 1)
            docs_score[doc_id] = docs_score.get(doc_id, 0) + rrf 
            
            logger.info(f"向量检索结果: 文档{idx + 1}: RRF分数: {rrf:.4f}")

        # 计算BM25检索结果的RRF分数
        for idx, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            docs_object[doc_id] = doc 
            
            rrf = 1 / (k + idx + 1)
            docs_score[doc_id] =  docs_score.get(doc_id, 0) + rrf   
            
            logger.info(f"BM25检索结果: 文档{idx + 1}: RRF分数: {rrf:.4f}")
            
        # 对rrf分数进行排序
        sorted_docs = sorted(docs_score.items(), key=lambda x:x[1], reverse=True)
        
        # 获取排序后文档
        reranked_docs = []
        for idx, score in sorted_docs:
            doc = docs_object[idx]
            doc.metadata['rrf_score'] = score
            reranked_docs.append(doc)
            logger.info(f"最终排序结果 - 文档{doc.page_content[:30]}... 最终分数{score:.4f}")
            
        logger.info(f"完成排序，共处理{len(reranked_docs)}个文档")
        return reranked_docs

def _format_result(doc: Document, idx: int) -> str:
    metadata = doc.metadata or {}
    name = metadata.get("name", "未知菜名")
    category = metadata.get("category", "未知分类")
    difficulty = metadata.get("difficulty", "未知难度")
    preview = doc.page_content.replace("\n", " ").strip()[:120]
    return f"[{idx}] {name} | {category} | {difficulty}\n  {preview}..."
    
if __name__ == "__main__":
    index_construction = IndexConstructionModule()
    data_preparation = DataPreparationModule(data_path="/workspaces/HowToCook/dishes")
    
    documents = data_preparation.load_documents()
    chunks = data_preparation.chunks_documents()
    parent_docs = data_preparation.get_parent_document(chunks)
    
    if not index_construction.load_index("./index"):
        index_construction.build_vector_index(chunks=chunks)
        index_construction.save_index()
            
    retrievalOptimization = RetrievalOptimizationModule(chunks=chunks, vectorstore=index_construction.vector_store)
    # results = retrievalOptimization.hybrid_retrieve("牛肉")
    filters = {"difficulty": "中等"}
    results = retrievalOptimization.metadata_filtered_search("牛肉", filters=filters, top_k=3)
    
    print(f"\n混合检索结果数量: {len(results)}")
    for i, result in enumerate(results):
        print(_format_result(result, i + 1))