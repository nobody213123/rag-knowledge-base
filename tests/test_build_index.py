"""
文档加载与分块测试
测试实际 app.rag.loader 中的 load_documents、split_documents
"""
import tempfile
import shutil
from pathlib import Path
from app.rag.loader import load_documents, split_documents


# ========== load_documents 测试 ==========

def test_load_documents_with_txt():
    tmpdir = tempfile.mkdtemp()
    try:
        txt_file = Path(tmpdir) / "test.txt"
        txt_file.write_text("测试内容", encoding="utf-8")
        docs = load_documents(Path(tmpdir))
        assert len(docs) == 1
        assert docs[0].page_content == "测试内容"
    finally:
        shutil.rmtree(tmpdir)


def test_load_documents_empty_dir():
    tmpdir = tempfile.mkdtemp()
    try:
        docs = load_documents(Path(tmpdir))
        assert docs == []
    finally:
        shutil.rmtree(tmpdir)


def test_load_documents_nonexistent_dir():
    docs = load_documents(Path("/nonexistent/path/xyz_123_test"))
    assert docs == []


def test_load_documents_multiple_files():
    tmpdir = tempfile.mkdtemp()
    try:
        (Path(tmpdir) / "a.txt").write_text("文件A", encoding="utf-8")
        (Path(tmpdir) / "b.txt").write_text("文件B", encoding="utf-8")
        docs = load_documents(Path(tmpdir))
        assert len(docs) == 2
    finally:
        shutil.rmtree(tmpdir)


def test_load_documents_skips_unsupported():
    tmpdir = tempfile.mkdtemp()
    try:
        (Path(tmpdir) / "data.csv").write_text("a,b,c", encoding="utf-8")
        (Path(tmpdir) / "readme.md").write_text("# Title", encoding="utf-8")
        docs = load_documents(Path(tmpdir))
        assert len(docs) == 0
    finally:
        shutil.rmtree(tmpdir)


# ========== split_documents 测试 ==========

def test_split_documents_basic():
    from langchain_core.documents import Document
    docs = [Document(page_content="A" * 300, metadata={"source": "test.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2


def test_split_documents_short_content():
    from langchain_core.documents import Document
    docs = [Document(page_content="短内容", metadata={"source": "test.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == "短内容"


def test_split_documents_empty():
    chunks = split_documents([], chunk_size=200, chunk_overlap=50)
    assert chunks == []


def test_split_documents_preserves_source():
    from langchain_core.documents import Document
    docs = [Document(page_content="A" * 500, metadata={"source": "my_doc.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    for chunk in chunks:
        assert chunk.metadata["source"] == "my_doc.txt"
