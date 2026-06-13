# API 文档

## 基础信息
- 基础 URL: `http://localhost:8000`
- API 版本: 2.1.0
- 数据格式: JSON
- 模型: Qwen-Plus (阿里云百炼)
- 检索: BM25 + 向量 MMR + RRF 融合

## 接口列表

### POST /chat/ask - 单次问答

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
    "sources": ["documents/p8_售后服务政策.txt"],
    "sources_detail": [
        {
            "index": 1,
            "file": "p8_售后服务政策.txt",
            "full_path": "documents/p8_售后服务政策.txt",
            "preview": "产品保修期为一年，自购买之日起计算..."
        }
    ],
    "retrieve_cost_ms": 110.3,
    "llm_cost_ms": 991.4,
    "total_cost_ms": 1102.5
}
```

**错误响应:**
- 400: 问题为空
- 500: 服务内部错误

---

### POST /chat/chat - 多轮对话

**请求体:**
```json
{
    "question": "那退货政策呢？",
    "session_id": "user-abc",
    "use_history": true
}
```

**响应体:**
```json
{
    "answer": "退货政策为7天无理由退货，需保持商品完好[2]。",
    "sources": ["documents/d19_订单发货与物流说明.txt"],
    "sources_detail": [
        {
            "index": 2,
            "file": "d19_订单发货与物流说明.txt",
            "full_path": "documents/d19_订单发货与物流说明.txt",
            "preview": "退货政策：7天无理由退货..."
        }
    ],
    "history_length": 2,
    "retrieve_cost_ms": 110.3,
    "llm_cost_ms": 991.4,
    "total_cost_ms": 1102.5
}
```

**参数说明:**
- `question`: 用户问题（必填）
- `session_id`: 会话 ID（可选，默认 "default"）
- `use_history`: 是否使用历史上下文（可选，默认 true）

**错误响应:**
- 400: 问题为空
- 500: 服务内部错误

---

### GET /chat/history - 获取对话历史

**查询参数:**
- `session_id`: 会话 ID（可选，默认 "default"）

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

### POST /chat/history/clear - 清空对话历史

**查询参数:**
- `session_id`: 会话 ID（可选，默认 "default"）

**响应体:**
```json
{
    "message": "会话 default 的历史已清空"
}
```

---

### GET /system/health - 健康检查

**响应体:**
```json
{
    "status": "healthy",
    "version": "2.1.0",
    "rag_engine_loaded": true
}
```

---

### GET /system/stats - 调用统计

**响应体:**
```json
{
    "total_queries": 100,
    "avg_retrieve_ms": 110.3,
    "avg_llm_ms": 991.4,
    "avg_total_ms": 1102.5
}
```

---

### POST /system/rebuild - 重建索引

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
# 设置 API Key
export DASHSCOPE_API_KEY=sk-xxx
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 构建索引
python -m scripts.build_index

# 启动服务
python -m app.main
```

服务启动后访问:
- 聊天界面: http://localhost:8000/
- API: http://localhost:8000/docs
- ReDoc 文档: http://localhost:8000/redoc
