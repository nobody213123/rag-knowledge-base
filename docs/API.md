# API 文档

## 基础信息
- 基础 URL: `http://localhost:8000`
- API 版本: 2.0.0
- 数据格式: JSON

## 接口列表

### POST /ask - 单次问答

**请求体:**
```json
{
    "question": "产品保修期多久？"
}
```

**响应体:**
```json
{
    "answer": "产品保修期为一年，自购买之日起计算[1]。",
    "sources": ["./documents/企业知识库.docx"],
    "溯源信息": [
        {
            "index": 1,
            "file": "企业知识库.docx",
            "full_path": "./documents/企业知识库.docx",
            "preview": "产品保修期为一年，自购买之日起计算..."
        }
    ],
    "retrieve_cost_ms": 35.2,
    "llm_cost_ms": 3200.5,
    "total_cost_ms": 3235.7
}
```

**错误响应:**
- 400: 问题为空
- 503: RAG 引擎未就绪

---

### POST /chat - 多轮对话

**请求体:**
```json
{
    "question": "那退货政策呢？",
    "use_history": true
}
```

**响应体:**
```json
{
    "answer": "退货政策为7天无理由退货，需保持商品完好[2]。",
    "sources": ["./documents/售后增值.txt"],
    "溯源信息": [
        {
            "index": 2,
            "file": "售后增值.txt",
            "full_path": "./documents/售后增值.txt",
            "preview": "退货政策：7天无理由退货..."
        }
    ],
    "history_length": 2,
    "retrieve_cost_ms": 32.1,
    "llm_cost_ms": 3100.3,
    "total_cost_ms": 3132.4
}
```

**参数说明:**
- `question`: 用户问题（必填）
- `use_history`: 是否使用历史上下文（可选，默认 true）

**错误响应:**
- 400: 问题为空
- 503: RAG 引擎未就绪

---

### GET /history - 获取对话历史

**响应体:**
```json
{
    "history": [
        {
            "question": "产品保修期多久？",
            "answer": "产品保修期为一年..."
        },
        {
            "question": "那退货政策呢？",
            "answer": "退货政策为7天无理由退货..."
        }
    ],
    "count": 2
}
```

---

### POST /history/clear - 清空对话历史

**响应体:**
```json
{
    "message": "对话历史已清空"
}
```

---

### GET /health - 健康检查

**响应体:**
```json
{
    "status": "healthy",
    "version": "2.0.0",
    "rag_engine_loaded": true
}
```

---

### GET /stats - 调用统计

**响应体:**
```json
{
    "total_queries": 100,
    "avg_retrieve_ms": 35.5,
    "avg_llm_ms": 3200.0,
    "avg_total_ms": 3235.5
}
```

---

### POST /rebuild - 重建索引

**响应体:**
```json
{
    "message": "索引重建完成"
}
```

---

### GET /docs - Swagger 文档

浏览器访问 `http://localhost:8000/docs` 查看交互式 API 文档。

---

## 启动服务

```bash
python app.py
```

服务启动后访问:
- API: http://localhost:8000
- Swagger 文档: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc
