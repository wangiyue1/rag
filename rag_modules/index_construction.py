import logging
from typing import List, Dict
from pathlib import Path

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from data_preparation import DataPreparationModule
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, force=True)

class IndexConstructionModule:
    """"
        1. 需要加载嵌入模型，同时构建向量索引
        2. 需提供保存和加载索引的功能，提高索引效率
    """
    def __init__(self, model_path: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./index"):
        self.model_path = model_path
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vector_store = None
        self._init_embedding_model()
        
    def _init_embedding_model(self):
        logger.info(f"初始化嵌入模型: {self.model_path}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_path,
            model_kwargs = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": False},
        )
        
        logger.info(f"初始化嵌入模型完成")

    def build_vector_index(self, chunks: List[Document]):
        logger.info(f"构建向量索引...")
        
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"向量索引构建完成，包含{len(chunks)}个文档")
        self.vector_store = vector_store

    def save_index(self):
        if self.vector_store is None:
            raise ValueError("向量索引尚未构建，无法保存")
    
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
        
    def load_index(self, index_path: str) -> bool:
        if self.embeddings is None:
            self._init_embedding_model()
        
        if not Path(index_path).exists():
            logger.warning(f"索引文件不存在: {index_path}")
            return False
    
        try:
            self.vector_store = FAISS.load_local(
                index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"成功加载索引: {index_path}")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {str(e)}")
            return False

    def similary_search(self, query: str) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("向量索引尚未构建或加载，无法执行相似度搜索")
        
        results = self.vector_store.similarity_search(query, k=3)
        return results


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
    
    results = index_construction.similary_search("如何烹饪鱼类菜肴？")
    print(f"\n检索结果数量: {len(results)}")
    for i, result in enumerate(results, start=1):
        print(_format_result(result, i))