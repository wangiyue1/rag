from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)


class RecipeRAGSystem:
    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        
        