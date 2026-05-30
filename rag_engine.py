"""
RAG 核心引擎
保留MMR检索 + 适配40条JSON测试集
修复：干扰题不计入召回率、分开统计业务题/拒答题
支持交互式问答 + eval批量评测
"""
import os
import time
import json
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# HuggingFace国内镜像 + 关闭警告
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载环境变量
load_dotenv()

# 基础配置
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "knowledge_base"

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

# 加载向量库
vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# 保留原版 MMR 配置
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

# 阿里云百炼客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    timeout=90.0
)

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )

def calc_recall(retrieved_sources, golden_sources):
    """
    计算单条召回率：支持 完整路径 包含 文件名 就算命中
    golden_sources: 标准答案文件名列表
    retrieved_sources: 检索返回的完整路径列表
    """
    hit = 0
    total = len(golden_sources)
    if total == 0:
        return None   # 干扰题，不参与召回率计算

    for gold_name in golden_sources:
        gold_name = gold_name.strip()
        # 只要检索到的任意一条路径 包含 目标文件名，就算命中
        for path in retrieved_sources:
            if gold_name in path:
                hit += 1
                break
    return hit / total

def ask(question: str):
    """单次问答：返回答案、检索文档、各阶段耗时"""
    total_start = time.time()

    # 检索耗时
    retrieve_start = time.time()
    docs = retriever.invoke(question)
    retrieve_cost = round((time.time() - retrieve_start) * 1000, 2)

    retrieved_sources = [doc.metadata.get("source", "") for doc in docs]
    context = format_docs(docs)

    # 提示词
    sys_prompt = """你是专业知识库助手，严格只根据参考资料回答，禁止编造。
规则：
1. 只能使用参考资料里的内容
2. 查不到相关资料就如实说明“暂无相关信息”，不要瞎编
3. 回答简洁通顺、用中文
参考资料：
{context}""".format(context=context)

    # LLM耗时
    llm_start = time.time()
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    llm_cost = round((time.time() - llm_start) * 1000, 2)
    total_cost = round((time.time() - total_start) * 1000, 2)

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved_sources,
        "retrieve_cost_ms": retrieve_cost,
        "llm_cost_ms": llm_cost,
        "total_cost_ms": total_cost
    }

def run_evaluation():
    """加载test_set.json，分开评测：业务题召回率 / 干扰题拒答"""
    print("\n========== 开始批量 RAG 评测 ==========")
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 业务题（有相关文档）
    recall_list = []
    # 全部题耗时
    retrieve_time_list = []
    llm_time_list = []
    total_time_list = []
    # 干扰题计数
    disturb_count = 0
    refuse_correct = 0

    for item in test_data:
        q = item["question"]
        golden_doc = [item["related_doc"].strip()]
        is_disturb = (item["related_doc"].strip() == "")

        res = ask(q)
        recall = calc_recall(res["retrieved_sources"], golden_doc)

        retrieve_time_list.append(res["retrieve_cost_ms"])
        llm_time_list.append(res["llm_cost_ms"])
        total_time_list.append(res["total_cost_ms"])

        print(f"\n【问题】{q}")
        if is_disturb:
            # 干扰题
            disturb_count += 1
            # 判断是否正确拒答
            if "暂无相关信息" in res["answer"] or "没有相关" in res["answer"]:
                refuse_correct += 1
                print("本条类型：干扰题 | 拒答正确")
            else:
                print("本条类型：干扰题 | 拒答失败(编造答案)")
        else:
            # 业务题
            recall_list.append(recall)
            print(f"本条召回率：{recall:.2%}")

        print(f"检索耗时：{res['retrieve_cost_ms']} ms | LLM耗时：{res['llm_cost_ms']} ms | 总耗时：{res['total_cost_ms']} ms")

    # 平均值计算
    avg_retrieve = sum(retrieve_time_list) / len(retrieve_time_list)
    avg_llm = sum(llm_time_list) / len(llm_time_list)
    avg_total = sum(total_time_list) / len(total_time_list)

    print("\n========== 评测汇总结果 ==========")
    if recall_list:
        avg_recall = sum(recall_list) / len(recall_list)
        print(f"业务题总数：{len(recall_list)} 道")
        print(f"业务题平均召回率：{avg_recall:.2%}")
    print(f"干扰题总数：{disturb_count} 道")
    if disturb_count > 0:
        print(f"拒答准确率：{refuse_correct / disturb_count:.2%}")
    print(f"全局平均检索耗时：{avg_retrieve:.2f} ms")
    print(f"全局平均LLM耗时：{avg_llm:.2f} ms")
    print(f"全局平均总响应耗时：{avg_total:.2f} ms")
    print("==================================\n")

if __name__ == "__main__":
    # 预热向量库
    print("正在预热向量库...")
    _ = retriever.invoke("预热")
    print("RAG 引擎已启动")
    print("输入 eval = 批量评测召回率&平均耗时&拒答率")
    print("直接输问题 = 单次问答")
    print("输入 quit 退出")
    print("-" * 40)

    while True:
        user_in = input("\n请输入：").strip()
        if user_in.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if user_in.lower() == "eval":
            run_evaluation()
            continue
        if not user_in:
            continue

        res = ask(user_in)
        print(f"\n回答：{res['answer']}")
        print("\n参考来源：")
        for src in res["retrieved_sources"]:
            print(f"- {src}")
        print(f"【性能】检索 {res['retrieve_cost_ms']}ms | LLM {res['llm_cost_ms']}ms | 总耗时 {res['total_cost_ms']}ms")