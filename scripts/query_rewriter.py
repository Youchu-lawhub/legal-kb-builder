#!/usr/bin/env python3
"""
查询改写模块（v1）

在检索前对用户查询做改写，提升召回率和精度：
  1. 同义词扩展（规则模式，无 LLM 依赖）
  2. 指代消解增强（比 consultation_agent 的简单拼接更强）
  3. 复杂问题拆解为子问题（LLM 模式，可选）

两种模式：
  A. 规则模式（默认，无外部依赖）：
     - jieba 分词 + 同义词表扩展
     - 法律术语规范化（如"打人"→"人身损害"）
     - 简单指代消解（基于上下文窗口）
     - 查询扩展（加同义词 OR 查询）

  B. LLM 模式（可选，需配置 API）：
     - 用 LLM 改写查询（更精准的语义改写）
     - 复杂问题拆解为多个子查询（多跳检索）
     - 需设置环境变量 LLM_API_KEY 和 LLM_API_BASE

使用方式：
    from query_rewriter import QueryRewriter

    # 规则模式（默认）
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("善意取得的要件")
    # -> {"original": "...", "rewritten": "...", "expansions": [...]}

    # LLM 模式
    rewriter = QueryRewriter(mode="llm")
    result = rewriter.rewrite("表见代理和无权代理的区别")
    # -> {"original": "...", "rewritten": "...", "sub_queries": [...]}
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试加载同义词扩展表
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# 尝试加载 jieba
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


# ─── 法律术语同义词表（内置，无需外部文件） ───
LEGAL_SYNONYMS = {
    # 物权
    "善意取得": ["善意受让", "善意第三人"],
    "善意": ["不知情", "非明知"],
    "恶意": ["明知", "应知"],
    "抵押": ["担保", "质押"],
    "无权处分": ["无权处置", "擅自处分"],
    # 合同
    "违约": ["违反合同", "不履行"],
    "解除合同": ["终止合同", "解除"],
    "损害赔偿": ["赔偿损失", "赔偿"],
    "可得利益": ["预期利益", "履行利益"],
    "格式条款": ["标准条款", "附合条款"],
    # 侵权
    "侵权": ["侵害", "侵害权益"],
    "人身损害": ["人身伤害", "打人"],
    "过错": ["过失", "故意"],
    "无过错": ["无过失"],
    # 总则
    "表见代理": ["表见", "权利外观代理"],
    "无权代理": ["越权代理", "无代理权"],
    "诉讼时效": ["时效", "消灭时效"],
    "撤销": ["撤回", "取消"],
    # 继承
    "遗嘱": ["遗言", "遗嘱继承"],
    "法定继承": ["法定遗产继承"],
    # 担保
    "保证": ["担保", "保证人"],
    "抵押权": ["担保物权", "抵押"],
    # 婚姻
    "离婚": ["婚姻关系解除", "解除婚姻"],
    "抚养权": ["子女抚养", "监护权"],
    # 程序
    "起诉": ["提起诉讼", "诉讼"],
    "上诉": ["提起上诉"],
    "执行": ["强制执行", "执行程序"],
}

# 法律术语规范化（口语→书面语）
LEGAL_NORMALIZE = {
    "打人": "人身损害",
    "被打": "人身损害",
    "借钱不还": "借款合同违约",
    "欠钱不还": "借款合同违约",
    "骗钱": "诈骗",
    "偷东西": "盗窃",
    "杀人": "故意杀人",
    "离婚官司": "离婚纠纷",
    "房产纠纷": "房屋物权纠纷",
    "遗产": "继承财产",
    "打官司": "诉讼",
    "告": "起诉",
}

# 指代词列表
COREF_WORDS = [
    "这个", "那个", "它", "该", "上述", "前面提到",
    "刚才说的", "这种情况", "这种", "那个问题", "这件事"
]


class QueryRewriter:
    """
    查询改写器

    Args:
        mode: "rule"（默认，规则模式）或 "llm"（LLM 模式）
        synonym_path: 自定义同义词表 YAML 路径（可选，覆盖内置表）
        llm_api_key: LLM API Key（LLM 模式用，默认从环境变量读取）
        llm_api_base: LLM API 基础 URL（LLM 模式用）
    """

    def __init__(
        self,
        mode: str = "rule",
        synonym_path: str = None,
        llm_api_key: str = None,
        llm_api_base: str = None,
    ):
        self._mode = mode
        self._synonyms = dict(LEGAL_SYNONYMS)
        self._normalize = dict(LEGAL_NORMALIZE)

        # 加载自定义同义词表（覆盖内置）
        if synonym_path and HAS_YAML:
            try:
                with open(synonym_path, 'r', encoding='utf-8') as f:
                    custom = yaml.safe_load(f) or {}
                if 'synonyms' in custom:
                    self._synonyms.update(custom['synonyms'])
                if 'normalize' in custom:
                    self._normalize.update(custom['normalize'])
            except Exception as e:
                print(f"[QueryRewriter] 加载同义词表失败: {e}")

        # LLM 配置
        self._llm_api_key = llm_api_key or os.environ.get('LLM_API_KEY', '')
        self._llm_api_base = llm_api_base or os.environ.get(
            'LLM_API_BASE', 'https://api.openai.com/v1'
        )

    def rewrite(
        self,
        query: str,
        context: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        改写查询。

        Args:
            query: 原始查询
            context: 对话上下文（用于指代消解）

        Returns:
            {
                "original": 原始查询,
                "rewritten": 改写后查询,
                "expansions": [同义词扩展列表],  # 规则模式
                "sub_queries": [子查询列表],      # LLM 模式（复杂问题拆解）
                "mode": 使用的模式,
            }
        """
        if self._mode == "llm" and self._llm_api_key:
            return self._rewrite_with_llm(query, context)
        return self._rewrite_with_rules(query, context)

    # ─── 规则模式 ───

    def _rewrite_with_rules(
        self,
        query: str,
        context: List[Dict] = None,
    ) -> Dict[str, Any]:
        """规则模式改写：规范化 + 同义词扩展 + 指代消解"""
        rewritten = query
        expansions = []

        # Step 1: 口语→书面语规范化
        for colloquial, formal in self._normalize.items():
            if colloquial in rewritten:
                rewritten = rewritten.replace(colloquial, formal)

        # Step 2: 指代消解（基于上下文）
        if context and self._has_coreference(rewritten):
            rewritten = self._resolve_coreference(rewritten, context)

        # Step 3: 同义词扩展（提取关键词的同义词）
        keywords = self._extract_keywords(rewritten)
        for kw in keywords:
            if kw in self._synonyms:
                expansions.extend(self._synonyms[kw])

        # 去重
        expansions = list(dict.fromkeys(expansions))

        return {
            "original": query,
            "rewritten": rewritten,
            "expansions": expansions,
            "mode": "rule",
        }

    def _has_coreference(self, query: str) -> bool:
        """检测是否含指代词"""
        return any(w in query for w in COREF_WORDS)

    def _resolve_coreference(
        self, query: str, history: List[Dict]
    ) -> str:
        """
        指代消解（增强版）。

        比 consultation_agent 的简单拼接更强：
        - 提取上一轮的问题主题（取名词性关键词）
        - 用主题替换指代词，而非整句拼接
        """
        if not history:
            return query

        last = history[-1]
        last_q = last.get("resolved", last.get("question", ""))

        # 提取上一轮的关键法律术语
        last_keywords = self._extract_keywords(last_q)
        if not last_keywords:
            # 降级为简单拼接
            return f"关于「{last_q}」的进一步问题：{query}"

        # 用最重要的关键词补充当前查询
        main_topic = last_keywords[0]  # 第一个关键词作为主题

        # 替换指代词
        result = query
        for coref in COREF_WORDS:
            if coref in result:
                result = result.replace(coref, main_topic)
                return result

        # 无明确指代词但上下文相关，前置主题
        return f"{main_topic} {query}"

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（用 jieba 或简单分词）"""
        if HAS_JIEBA:
            words = list(jieba.cut(text, cut_all=False))
            # 过滤停用词和短词
            stop_words = {"的", "了", "是", "在", "有", "和", "与", "及", "或", "等"}
            return [w for w in words if len(w) >= 2 and w not in stop_words]
        # 降级：按空格和标点分词
        return [w for w in re.split(r'[\s，。、？]+', text) if len(w) >= 2]

    # ─── LLM 模式 ───

    def _rewrite_with_llm(
        self,
        query: str,
        context: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        LLM 模式改写：用 LLM 做语义改写 + 子问题拆解。

        需要环境变量 LLM_API_KEY。依赖不可用时降级为规则模式。
        """
        try:
            import urllib.request
            import json as json_mod
        except ImportError:
            return self._rewrite_with_rules(query, context)

        # 构造 prompt
        context_str = ""
        if context:
            last = context[-1]
            context_str = f"\n上一轮对话: {last.get('question', '')}\n"

        prompt = f"""你是一个法律查询改写助手。请改写以下查询，使其更适合法律知识库检索。

原始查询: {query}{context_str}

请返回 JSON 格式：
{{
  "rewritten": "改写后的查询（使用规范法律术语，消除指代）",
  "sub_queries": ["子查询1", "子查询2"]
}}

要求：
1. rewritten: 用规范法律术语重写，消除口语和指代
2. sub_queries: 若是复杂问题（涉及多个法律概念），拆解为2-3个独立子查询；简单问题返回空数组
3. 只返回 JSON，不要其他文字"""

        # 调用 LLM API
        try:
            req_data = json_mod.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }).encode('utf-8')

            req = urllib.request.Request(
                f"{self._llm_api_base}/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._llm_api_key}",
                },
                method='POST',
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json_mod.loads(resp.read().decode('utf-8'))

            content = result['choices'][0]['message']['content']
            # 解析 JSON（兼容 markdown 代码块）
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            parsed = json_mod.loads(content)

            return {
                "original": query,
                "rewritten": parsed.get("rewritten", query),
                "sub_queries": parsed.get("sub_queries", []),
                "mode": "llm",
            }

        except Exception as e:
            print(f"[QueryRewriter] LLM 改写失败，降级为规则模式: {e}")
            return self._rewrite_with_rules(query, context)


# ─── 全局单例 ───

import threading

_rewriter_instance = None
_rewriter_lock = threading.Lock()


def get_rewriter(mode: str = "rule") -> QueryRewriter:
    """获取全局 QueryRewriter 单例（线程安全）"""
    global _rewriter_instance
    if _rewriter_instance is None:
        with _rewriter_lock:
            if _rewriter_instance is None:
                _rewriter_instance = QueryRewriter(mode=mode)
    return _rewriter_instance


# ─── CLI ───

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python query_rewriter.py --test                    # 自测")
        print("  python query_rewriter.py <query>                   # 规则模式改写")
        print("  python query_rewriter.py <query> --llm             # LLM 模式改写")
        sys.exit(0)

    if sys.argv[1] == '--test':
        r = QueryRewriter(mode="rule")
        tests = [
            "善意取得的要件",
            "打人要赔多少钱",
            "表见代理和无权代理的区别",
            "这个怎么处理",
        ]
        print("规则模式改写测试:")
        for q in tests:
            result = r.rewrite(q)
            print(f"\n  原始: {q}")
            print(f"  改写: {result['rewritten']}")
            print(f"  扩展: {result['expansions']}")
        print("\n可用同义词:", len(LEGAL_SYNONYMS), "组")
        print("可用规范化:", len(LEGAL_NORMALIZE), "条")
        sys.exit(0)

    # 单查询改写
    mode = "rule"
    query = sys.argv[1]
    if "--llm" in sys.argv:
        mode = "llm"

    r = QueryRewriter(mode=mode)
    result = r.rewrite(query)
    print(f"原始: {result['original']}")
    print(f"改写: {result['rewritten']}")
    if result.get('expansions'):
        print(f"扩展: {result['expansions']}")
    if result.get('sub_queries'):
        print(f"子查询: {result['sub_queries']}")
    print(f"模式: {result['mode']}")
