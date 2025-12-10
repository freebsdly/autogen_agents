#!/usr/bin/env python3
"""
Rust文档向量化工具
将Rust HTML文档解析、切分并向量化存储到Qdrant数据库
"""

from math import log
import os
import glob
import re
import logging
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm

import nltk
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Text
from dynaconf import Dynaconf, Validator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rust_docs_vectorization.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class RustDocsVectorizer:
    """
    Rust文档向量化器，负责解析HTML文档、切分文本并向量化存储
    """
    
    def __init__(self, config_path: str = 'config.ini'):
        """
        初始化向量化器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.client = None
        self.model = None
        self._initialize_tools()
        
    def _load_config(self, config_path: str) -> Dynaconf:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
        """
        # 检查配置文件是否存在
        config_file = config_path if os.path.exists(config_path) else None
        
        # 创建Dynaconf配置对象
        config = Dynaconf(
            settings_files=[config_file] if config_file else None,
            # 默认配置
            default_settings={
                'qdrant_url': 'http://localhost:6333',
                'embedding_model': 'all-MiniLM-L6-v2',
                'collection_name': 'rust_docs_embeddings',
                'vector_size': 384,  # all-MiniLM-L6-v2的向量维度
                'max_chunk_length': 1024,
                'chunk_overlap': 100,
                'min_chunk_size': 50,
                'batch_size': 100
            },
            # 验证配置
            validators=[
                Validator('qdrant_url', must_exist=True, is_type_of=str),
                Validator('embedding_model', must_exist=True, is_type_of=str),
                Validator('collection_name', must_exist=True, is_type_of=str),
                Validator('vector_size', must_exist=True, is_type_of=int),
                Validator('max_chunk_length', must_exist=True, is_type_of=int),
                Validator('chunk_overlap', must_exist=True, is_type_of=int),
                Validator('min_chunk_size', must_exist=True, is_type_of=int),
                Validator('batch_size', must_exist=True, is_type_of=int),
            ]
        )
        
        if config_file:
            logger.info(f"已加载配置文件: {config_path}")
        else:
            logger.warning(f"未找到配置文件 {config_path}，使用默认配置")
        
        return config
    
    def _initialize_tools(self):
        """
        初始化工具组件（NLTK、Qdrant客户端、Embedding模型）
        """
        # 初始化NLTK
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt模块已下载")
        except Exception as e:
            logger.error(f"NLTK初始化失败: {e}")
            raise
        
        # 初始化Qdrant客户端
        try:
            self.client = QdrantClient(url=self.config.qdrant_url)
            
            # 验证Qdrant服务器连接
            try:
                self.client.get_collections()
                logger.info(f"已连接到Qdrant服务器: {self.config.qdrant_url}")
            except Exception as ping_error:
                logger.error(f"Qdrant服务器连接验证失败: {ping_error}")
                self.client = None
                raise RuntimeError(f"无法连接到Qdrant服务器: {ping_error}")
                
        except Exception as e:
            logger.error(f"Qdrant客户端初始化失败: {e}")
            self.client = None
            raise
        
        # 初始化Embedding模型
        try:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model, 
                device="cpu"
            )
            logger.info(f"已加载Embedding模型: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Embedding模型加载失败: {e}")
            self.embedding_model = None
            raise
    
    def extract_text_from_html(self, file_path: str) -> str:
        """
        从HTML文件中提取纯文本
        
        Args:
            file_path: HTML文件路径
            
        Returns:
            提取的纯文本
        """
        try:
            elements = partition_html(filename=file_path)
            text = '\n'.join([element.text for element in elements if isinstance(element, Text)])
            text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
            logger.debug(f"从文件 {file_path} 提取了 {len(text)} 个字符")
            return text
        except Exception as e:
            logger.error(f"解析HTML文件 {file_path} 时出错: {e}")
            return ""
    
    def split_text(self, text: str, max_length: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        将长文本切分为多个重叠的文本块，保留代码块
        
        Args:
            text: 原始文本
            max_length: 每个文本块的最大长度
            overlap: 相邻文本块的重叠长度
            
        Returns:
            切分后的文本块列表
        """
        if max_length is None:
            max_length = int(self.config.max_chunk_length)
        
        if overlap is None:
            overlap = int(self.config.chunk_overlap)
        
        if len(text) <= max_length:
            return [text]
        
        # 1. 识别并提取代码块
        code_blocks = []
        code_pattern = re.compile(r'```(?:rust)?(.*?)```', re.DOTALL)
        
        def replace_code(match):
            code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(code_blocks)-1}__'
        
        text_with_placeholders = code_pattern.sub(replace_code, text)
        
        # 2. 使用nltk按语义切分句子
        sentences = nltk.sent_tokenize(text_with_placeholders)
        
        # 3. 将句子组合成符合长度要求的文本块
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length + 1 <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # 开始新块，考虑重叠
                current_chunk = [sentence]
                current_length = sentence_length + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # 4. 将占位符替换回原始代码块
        final_chunks = []
        for chunk in chunks:
            for i, code_block in enumerate(code_blocks):
                chunk = chunk.replace(f'__CODE_BLOCK_{i}__', code_block)
            final_chunks.append(chunk)
        
        return final_chunks
    
    def process_html_files(self, docs_dir: str, file_filter: Optional[Callable[[str], bool]] = None) -> List[Dict[str, Any]]:
        """
        处理指定目录下的所有HTML文件
        
        Args:
            docs_dir: 文档目录路径
            file_filter: 可选的自定义文件过滤函数，接收文件路径并返回布尔值，返回True的文件将被保留
            
        Returns:
            处理后的文档块列表
        """
        if not os.path.exists(docs_dir):
            logger.error(f"文档目录不存在: {docs_dir}")
            return []
        
        html_files = glob.glob(os.path.join(docs_dir, '**', '*.html'), recursive=True)
        
        # 应用文件过滤器
        if file_filter:
            filtered_files = []
            for file_path in html_files:
                if file_filter(file_path):
                    filtered_files.append(file_path)
                else:
                    logger.debug(f"文件被过滤: {file_path}")
            html_files = filtered_files
            logger.info(f"应用过滤器后，剩余 {len(html_files)} 个HTML文件")
        else:
            logger.info(f"找到 {len(html_files)} 个HTML文件")
        
        documents = []
        min_chunk_size = int(self.config.min_chunk_size)
        
        for file_path in tqdm(html_files, desc="处理HTML文件"):
            try:
                text = self.extract_text_from_html(file_path)
                if text:
                    chunks = self.split_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk) > min_chunk_size:
                            documents.append({
                                'file_path': file_path,
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'text': chunk
                            })
                    
                    logger.debug(f"已处理: {file_path} (切分为 {len(chunks)} 个块)")
                else:
                    logger.warning(f"文件内容为空: {file_path}")
            except Exception as e:
                logger.error(f"处理文件时出错 {file_path}: {e}")
        
        logger.info(f"共处理 {len(documents)} 个文本块")
        return documents
    
    def create_collection(self):
        """
        创建Qdrant集合（如果不存在）
        """
        # 检查Qdrant客户端是否已初始化
        if self.client is None:
            logger.error("Qdrant客户端未初始化")
            raise RuntimeError("Qdrant客户端未初始化，请检查Qdrant服务器连接")
        
        collection_name = self.config.collection_name
        vector_size = int(self.config.vector_size)
        
        try:
            collections = self.client.get_collections().collections
            if not any(col.name == collection_name for col in collections):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"已创建集合: {collection_name}")
            else:
                logger.info(f"集合已存在: {collection_name}")
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
    
    def vectorize_and_store(self, documents: List[Dict[str, Any]]):
        """
        将文档向量化并存储到Qdrant数据库
        
        Args:
            documents: 文档块列表
        """
        if not documents:
            logger.error("没有要处理的文档")
            return
        
        # 检查Qdrant客户端是否已初始化
        if self.client is None:
            logger.error("Qdrant客户端未初始化")
            raise RuntimeError("Qdrant客户端未初始化，请检查Qdrant服务器连接")
        
        # 检查Embedding模型是否已初始化
        if self.embedding_model is None:
            logger.error("Embedding模型未初始化")
            raise RuntimeError("Embedding模型未初始化，请检查模型加载是否成功")
            
        # 创建集合
        self.create_collection()
        
        # 提取文本
        texts = [doc['text'] for doc in documents]
        logger.info(f"开始向量化 {len(texts)} 个文本块")
        
        # 向量化
        try:
            embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            logger.info(f"向量化完成，生成 {embeddings.shape[0]} 个向量")
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            raise
        
        # 准备数据点
        points = []
        for i, doc in enumerate(documents):
            point = models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": doc['text'],
                    "file_path": doc['file_path'],
                    "chunk_index": doc['chunk_index'],
                    "total_chunks": doc['total_chunks']
                }
            )
            points.append(point)
        
        # 批量插入
        collection_name = self.config.collection_name
        batch_size = int(self.config.batch_size)
        
        logger.info(f"开始批量插入向量，批次大小: {batch_size}")
        
        for i in tqdm(range(0, len(points), batch_size), desc="插入向量"):
            batch = points[i:i+batch_size]
            try:
                self.client.upsert(collection_name=collection_name, points=batch)
                logger.debug(f"已插入批次 {i//batch_size + 1}，共 {len(batch)} 个向量")
            except Exception as e:
                logger.error(f"插入批次 {i//batch_size + 1} 失败: {e}")
                raise
        
        logger.info(f"向量已成功存入Qdrant集合 {collection_name}")
        logger.info(f"共存入 {len(points)} 个文档向量")


def main():
    """
    主函数
    """
    import argparse
    
    # 定义按文件名过滤的函数，保留文件名以"ch[0-9][0-9]-[0-9][0-9]"开头的文件
    def filter_chapter_files(file_path: str) -> bool:
        """
        过滤函数，保留文件名以"ch[0-9][0-9]-[0-9][0-9]"开头的文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果文件名以"ch[0-9][0-9]-[0-9][0-9]"开头则返回True，否则返回False
        """
        file_name = os.path.basename(file_path)
        return re.match(r'^ch\d{2}-\d{2}', file_name) is not None
    
    parser = argparse.ArgumentParser(description='Rust文档向量化工具')
    parser.add_argument('docs_dir', help='Rust文档目录路径')
    parser.add_argument('-c', '--config', default='config.toml', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        vectorizer = RustDocsVectorizer(args.config)
        documents = vectorizer.process_html_files(args.docs_dir, file_filter=filter_chapter_files)
        
        if not documents:
            logger.error("没有找到有效的HTML文件或提取的文本为空")
            return 1
        
        vectorizer.vectorize_and_store(documents)
        logger.info("文档向量化任务完成")
        return 0
        
    except KeyboardInterrupt:
        logger.info("任务被用户中断")
        return 130
    except Exception as e:
        logger.error(f"任务执行失败: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())