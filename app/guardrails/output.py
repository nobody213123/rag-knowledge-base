"""
输出 Guardrail：PII 检测 + 脱敏

只替换不阻断，命中后记录审计日志
"""
import re
from app.logger import get_logger

logger = get_logger("guardrails.output")

PII_PATTERNS = [
    # 中国大陆手机号
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), lambda m: m.group()[:3] + "****" + m.group()[-4:]),
    # 中国大陆身份证号（15 或 18 位）
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), lambda _: "*****************"),
    (re.compile(r"(?<!\d)\d{15}(?!\d)"), lambda _: "***************"),
    # 银行卡号（16-19 位数字）
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), lambda m: m.group()[:6] + "********" + m.group()[-4:]),
    # 中国大陆固定电话（带区号）
    (re.compile(r"0\d{2,3}-?\d{7,8}"), lambda m: m.group().split("-")[0] + "-*******"),
    # 邮箱地址
    (re.compile(r"[\w.]+@\w+\.\w+"), lambda m: m.group().split("@")[0][:2] + "****@" + m.group().split("@")[1]),
]

_PII_JUDGE_PROMPT = """检查以下文本是否包含明显的敏感个人信息（手机号、身份证、银行卡）。
如果包含，回答"有"；否则回答"无"。
文本：{text}"""


def mask_pii(text: str) -> tuple[str, list[str]]:
    """对文本中所有 PII 做脱敏处理，返回 (脱敏后的文本, 发现的PII类型列表)"""
    found_types = []
    original = text

    for pattern, replacer in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # 记录类型
            type_name = _infer_type(pattern)
            found_types.append(type_name)

            # 替换
            text = pattern.sub(replacer, text)

    if found_types:
        logger.info(f"PII 脱敏: {', '.join(found_types)} | 原文: {original[:40]}...")

    return text, found_types


async def check_output(text: str) -> tuple[str, bool]:
    """
    输出安全检测
    返回 (处理后的文本, 是否有安全问题)

    步骤：
      ① PII 正则脱敏
      ② LLM 安全巡检（仅当有 PII 命中时触发，防止遗漏）
    """
    masked, found_types = mask_pii(text)
    has_issues = len(found_types) > 0

    if has_issues:
        try:
            from app.model.registry import get_model_registry
            messages = [
                {"role": "system",
                 "content": _PII_JUDGE_PROMPT.format(text=masked[:200])},
                {"role": "user", "content": "请检查。"},
            ]
            verdict, _ = await get_model_registry().generate("judge", messages)
            if verdict.strip().startswith("无"):
                has_issues = False
        except Exception:
            pass

    return masked, has_issues


def _infer_type(pattern: re.Pattern) -> str:
    label_map = {
        r"1[3-9]": "手机号",
        r"\d{17}": "身份证",
        r"\d{15}": "身份证",
        r"\d{16,19}": "银行卡",
        r"0\d{2,3}": "固话",
        "@": "邮箱",
    }
    for key, label in label_map.items():
        if key in pattern.pattern:
            return label
    return "未知PII"
