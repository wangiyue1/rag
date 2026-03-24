from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class RAGConfig:

    data_path: str = "/workspaces/HowToCook/dishes"
    index_save_path: str = "./index"
    
    embedding_model: str ="BAAI/bge-small-zh-v1.5" 
    llm_model: str = "glm-4.7-flash-free"
        
    temperature: float = 0.1
    max_tokens: int = 2048
    top_k: int = 5

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return{
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k
        }
# 默认配置实例
DEFAULT_CONFIG = RAGConfig()