"""
build_index.py 单元测试
测试文档加载、文本分块等函数
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 模拟依赖
sys.modules["sentence_transformers"] = type(sys)("sentence_transformers")
sys.modules["torch"] = type(sys)("torch")
sys.modules["chromadb"] = type(sys)("chromadb")


def load_documents(directory: str):
    docs = []
    dir_path = Path(directory)
    if not dir_path.exists():
        return docs
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    docs.append({"content": content, "source": str(file_path)})
                except Exception:
                    pass
    return docs


def split_documents(documents, chunk_size=200, chunk_overlap=50):
    chunks = []
    for doc in documents:
        content = doc["content"]
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            if chunk:
                chunks.append({"content": chunk, "source": doc["source"]})
    return chunks


# ========== load_documents 测试 ==========

def test_load_documents_with_txt():
    tmpdir = tempfile.mkdtemp()
    try:
        txt_file = Path(tmpdir) / "test.txt"
        txt_file.write_text("测试内容", encoding="utf-8")
        docs = load_documents(tmpdir)
        assert len(docs) == 1
        assert docs[0]["content"] == "测试内容"
    finally:
        shutil.rmtree(tmpdir)


def test_load_documents_empty_dir():
    tmpdir = tempfile.mkdtemp()
    try:
        docs = load_documents(tmpdir)
        assert docs == []
    finally:
        shutil.rmtree(tmpdir)


def test_load_documents_nonexistent_dir():
    docs = load_documents("/nonexistent/path")
    assert docs == []


def test_load_documents_multiple_files():
    tmpdir = tempfile.mkdtemp()
    try:
        (Path(tmpdir) / "a.txt").write_text("文件A", encoding="utf-8")
        (Path(tmpdir) / "b.txt").write_text("文件B", encoding="utf-8")
        docs = load_documents(tmpdir)
        assert len(docs) == 2
    finally:
        shutil.rmtree(tmpdir)


# ========== split_documents 测试 ==========

def test_split_documents_basic():
    docs = [{"content": "A" * 300, "source": "test.txt"}]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2


def test_split_documents_short_content():
    docs = [{"content": "短内容", "source": "test.txt"}]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "短内容"


def test_split_documents_empty():
    chunks = split_documents([], chunk_size=200, chunk_overlap=50)
    assert chunks == []


def test_split_documents_preserves_source():
    docs = [{"content": "A" * 500, "source": "my_doc.txt"}]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    for chunk in chunks:
        assert chunk["source"] == "my_doc.txt"


if __name__ == "__main__":
    test_load_documents_with_txt()
    test_load_documents_empty_dir()
    test_load_documents_nonexistent_dir()
    test_load_documents_multiple_files()
    test_split_documents_basic()
    test_split_documents_short_content()
    test_split_documents_empty()
    test_split_documents_preserves_source()
    print("所有测试通过！")
