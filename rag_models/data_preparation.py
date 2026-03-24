import logging
from typing import List, Dict, Any

from pathlib import Path
import hashlib
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, force=True)
class DataPreparationModule:
    """
        数据准备模块：
        负责从指定路径加载文档，
        增强文档的元数据，
        并将文档切分成更小的片段以供后续处理,
        对文档去重
    """
    CATEGORY_MAPPING = {
        'meat_fish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(CATEGORY_MAPPING.values())
    DIFFICULTY_LABELS =  ['非常简单', '简单', '中等', '困难', '非常困难']
    def __init__(self, data_path: str = "/workspaces/HowToCook/dishes"):
        self.data_path = data_path
        self.parent_documents: List[Document] = []
        self.children_documents: List[Document] = []
        self.documents_pair: Dict[str, str] = {}

    def load_documents(self)-> List[Document]:
        logger.info("从路径加载文档: %s", self.data_path)
        
        data_path_obj = Path(self.data_path)
        documents = []
        
        # 递归查找所有.md文件，并加载内容
        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 使用相对路径确定唯一路径
                data_root = data_path_obj.resolve()
                relative_path = md_file.resolve().relative_to(data_root).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": md_file.as_posix(),
                        "parent_id": parent_id,
                        "type": "parent"
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"加载文档失败: {md_file.as_posix()}, 错误: {str(e)}")
        
        for doc in documents:
            self._enahnce_metadata(doc)
        
        self.parent_documents = documents
        logger.info(f"成功加载文档数量: {len(self.parent_documents)}") 
        return self.parent_documents
        
    def _enahnce_metadata(self, doc: Document):
        """增强文档的元数据：种类，菜名, 难度"""
        source = Path(doc.metadata.get('source', ''))
        parts = source.parts
        doc.metadata['category'] = '其他'
        
        for key, value in self.CATEGORY_MAPPING.items():
            if key in parts:
                doc.metadata['category'] = value
                break
        
        doc.metadata['name'] = source.stem
        
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'
    
    def chunks_documents(self)->List[Document]:
        chunks =self._markdown_split(self.parent_documents)
        
        # 进行兜底
        for idx, chunk in enumerate(chunks):
            if 'child_id' not in chunk.metadata:
                child_id = uuid.uuid4()
                chunk.metadata['child_id'] = str(child_id)
            chunk.metadata['batch_index'] = idx
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        self.children_documents = chunks
        return chunks
        
    def _markdown_split(self, documents: List[Document])-> List[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', "Header_1"),
                ('##', "Header_2"),
                ('###', "Header_3"),
            ],
            strip_headers=False,
        )
        
        all_chunks = []
        for document in documents:
            try:
                logger.info(f"开始切割文档{document.metadata.get('source', '未知来源')}")
                # 判断是否是Markdown文档
                content_preview = document.page_content[:200]
                has_header = any(line.strip().startswith(('#')) for line in content_preview.split('\n'))
                if not has_header:
                    logger.warning(f"文档可能不是Markdown格式，跳过切分: {document.metadata.get('source', '未知来源')}")
                    continue
                chunks = splitter.split_text(document.page_content)
                for idx, chunk in enumerate(chunks):
                    child_id = uuid.uuid4()
                    
                    chunk.metadata.update(document.metadata)
                    chunk.metadata.update({
                        "type": "child",
                        "child_id": str(child_id),
                        "chunk_index": idx
                    })
                    
                    self.documents_pair[child_id] = document.metadata['parent_id']
                    
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"文档切分失败: {document.metadata.get('source', '未知来源')}, 错误: {str(e)}")
        
        logger.info(f"成功切分文档数量: {len(all_chunks)}")
        return all_chunks
    
    def get_parent_document(self, child_chunks: List[Document])->List[Document]:
        """根据子文档获取对应的父文档, 进行去重"""
        parent_documents_map = {}
        parent_relevance = {}
        
        for chunk in child_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id:
                # 增加父文档的相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                
                # 获取父文档
                if parent_id not in parent_documents_map:
                    for doc in self.parent_documents:
                        if doc.metadata.get('parent_id') == parent_id:
                            parent_documents_map[parent_id] = doc
                            break
        
        # 根据相关性排序父文档
        sorted_parent_ids = sorted(parent_relevance.keys(), key=lambda x:parent_relevance[x], reverse=True)
        
        # 返回排序后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_documents_map:
                parent_docs.append(parent_documents_map[parent_id])
        
        # 打印日志
        parent_infos = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('name', '未知菜名')
            parent_id = doc.metadata.get('parent_id', '未知ID')
            relevance = parent_relevance.get(parent_id, 0)
            parent_infos.append(f"{dish_name} (ID: {parent_id}, 相关性: {relevance})")
        
        logger.info(f"从{len(child_chunks)}获取到{len(parent_docs)}个父文档: {', '.join(parent_infos)}")
        return parent_docs
    
if __name__ == "__main__":
    logger.info("开始数据准备模块测试")
    data_preparation = DataPreparationModule(data_path="/workspaces/HowToCook/dishes/aquatic")
    documents = data_preparation.load_documents()
    chunks = data_preparation.chunks_documents()
    parent_docs = data_preparation.get_parent_document(chunks)
    print(f"加载的父文档数量: {len(documents)}")
    print(f"切分后的子文档数量: {len(chunks)}")         