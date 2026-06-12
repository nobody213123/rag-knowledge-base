"""
文档加载与分块模块
负责读取 documents/ 目录下的文档并切分为 chunks
"""
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,     # PDF 解析
    Docx2txtLoader,  # Word 解析
    TextLoader,      # 纯文本解析
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.logger import get_logger

logger = get_logger("loader")


def load_documents(directory: Path = DOCUMENTS_DIR) -> list[Document]:
    """
    加载目录下所有支持的文档
    支持 .pdf / .docx / .txt 三种格式
    返回 langchain Document 对象列表
    """
    docs = []
    if not directory.exists():
        logger.warning(f"目录 {directory} 不存在，请创建并放入文档")
        return docs

    logger.info(f"开始扫描目录: {directory}")
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            try:
                # 根据文件后缀选择对应的加载器
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif suffix == ".docx":
                    loader = Docx2txtLoader(str(file_path))
                elif suffix == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                else:
                    logger.info(f"跳过不支持的文件类型: {file_path.name}")
                    continue

                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source"] = str(file_path)
                docs.extend(file_docs)
                logger.info(f"已加载: {file_path.name} ({len(file_docs)} 段)")

            except Exception as e:
                logger.error(f"加载失败 {file_path.name}: {e}")

    logger.info(f"共加载 {len(docs)} 个文档片段")
    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    将文档切分为小块（chunk）
    使用 RecursiveCharacterTextSplitter，按段落 → 句子 → 字符递进切割
    """
    logger.info(f"开始分块: chunk_size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"分块完成: {len(documents)} 段 -> {len(chunks)} 个文本块")
    return chunks
