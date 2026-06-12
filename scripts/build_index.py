"""
构建知识库索引脚本
用法: python -m scripts.build_index
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_documents, split_documents
from app.rag.retriever import build_vector_store
from app.logger import get_logger

logger = get_logger("build_index")


def main():
    logger.info("开始构建知识库索引")
    print("=" * 50)
    print("开始构建知识库索引")
    print("=" * 50)

    print("\n[步骤 1/3] 加载文档...")
    documents = load_documents()
    if not documents:
        logger.warning("未找到任何文档")
        print("未找到任何文档，请在 documents/ 目录放入文档后重试")
        return

    print("\n[步骤 2/3] 文本分块...")
    chunks = split_documents(documents)

    print("\n[步骤 3/3] 向量化存储...")
    build_vector_store(chunks)

    logger.info("知识库索引构建完成")
    print("\n" + "=" * 50)
    print("知识库索引构建完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
