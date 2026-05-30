"""
知识库索引构建脚本
功能：加载文档 -> 文本分块 -> 向量化存储到 Chroma
"""
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 全局配置
DOCUMENTS_DIR = "./documents"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def load_documents(directory: str):
    docs = []
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"警告：目录 {directory} 不存在，请创建并放入文档")
        return docs

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif suffix == ".docx":
                    loader = Docx2txtLoader(str(file_path))
                elif suffix == ".txt":
                    loader = TextLoader(
                        str(file_path),
                        encoding="utf-8",
                    )
                else:
                    print(f"跳过不支持的文件：{file_path.name}")
                    continue

                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source"] = str(file_path)
                docs.extend(file_docs)
                print(f"  已加载：{file_path.name} ({len(file_docs)} 段)")

            except Exception as e:
                print(f"  加载失败 {file_path.name}：{e}")

    print(f"共加载 {len(docs)} 个文档片段")
    return docs


def split_documents(documents, chunk_size=200, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"分块完成：{len(documents)} 段 -> {len(chunks)} 个文本块")
    return chunks


def build_vector_store(chunks, persist_dir):
    print(f"正在加载嵌入模型：{EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"正在构建向量库，存储到：{persist_dir}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="knowledge_base",
    )
    print(f"向量库构建完成，共 {len(chunks)} 个向量")
    return vector_store


def main():
    print("=" * 50)
    print("开始构建知识库索引")
    print("=" * 50)

    print("\n[步骤 1/3] 加载文档...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print("未找到任何文档，请在 documents/ 目录放入文档后重试")
        return

    print("\n[步骤 2/3] 文本分块...")
    chunks = split_documents(documents)

    print("\n[步骤 3/3] 向量化存储...")
    build_vector_store(chunks, CHROMA_PERSIST_DIR)

    print("\n" + "=" * 50)
    print("知识库索引构建完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()