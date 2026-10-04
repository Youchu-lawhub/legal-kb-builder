#!/usr/bin/env python3
"""
BM25 关键词检索模块

使用 jieba 中文分词 + rank_bm25 评分 + 法律同义词扩展，
提供比原始 keyword counting 准确得多的关键词检索能力。

使用方式：
    from bm25_searcher import BM25Searcher

    bm25 = BM25Searcher()

    # 构建索引（KB 创建时一次性执行）
    bm25.build_index(kb_path)

    # 查询
    results = bm25.search(kb_path, "善意取得的构成要件", top_k=20)
    # -> [(chunk_id, bm25_score), ...]
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shared_utils import load_synonym_expansion, atomic_write_json, file_signature

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# ─── 中文停用词 ───
STOPWORDS = {
    '的', '了', '是', '在', '和', '与', '或', '对', '把', '被',
    '让', '向', '从', '到', '给', '用', '以', '而', '但', '也',
    '都', '不', '没', '有', '这', '那', '就', '要', '会', '能',
    '可', '所', '其', '之', '上', '下', '中', '大', '小', '多',
    '少', '各', '每', '某', '本', '该', '此', '什么', '怎么',
    '如何', '为什么', '哪些', '哪个', '关于', '以及', '还是',
    '然而', '因此', '所以', '如果', '虽然', '已经', '正在',
    '可以', '应当', '应该', '必须', '需要', '进行', '通过',
    '根据', '按照', '依据', '对于', '作为', '由于', '即',
}


class BM25Searcher:
    """
    BM25 关键词检索器

    Args:
        synonym_path: 同义词扩展 YAML 路径（默认使用 legal-kb-agentic 的）
        custom_dict_path: jieba 自定义词典路径（可选）
    """

    def __init__(
        self,
        synonym_path: str = None,
        custom_dict_path: str = None,
    ):
        self._synonym_map = load_synonym_expansion(synonym_path)
        self._custom_dict_loaded = False
        self._custom_dict_path = custom_dict_path

        # 运行时缓存（避免重复加载同一 KB 的索引）
        self._cache: Dict[str, Any] = {}

        # 加载自定义词典
        if HAS_JIEBA:
            self._load_custom_dict()

    # ─── 构建时方法 ───

    def build_index(self, kb_path: Path, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        为指定知识库构建 BM25 索引。

        读取所有非索引 .md 文件，分块、分词，保存语料库和元数据。
        在 KB 创建后调用一次即可。

        Args:
            kb_path: 知识库目录路径
            chunk_size: 每块字符数（默认 512）
            chunk_overlap: 块间重叠字符数（默认 50）
        """
        if not HAS_JIEBA:
            raise ImportError("jieba 未安装。请运行: pip install jieba")
        if not HAS_BM25:
            raise ImportError("rank_bm25 未安装。请运行: pip install rank-bm25")

        kb_path = Path(kb_path)
        md_files = sorted(
            f for f in kb_path.glob("*.md")
            if not f.name.startswith("_")
        )

        if not md_files:
            print(f"[BM25] 未在 {kb_path} 找到 MD 文件")
            return

        all_chunks = []     # [{chunk_id, file, start_char, end_char, text_preview}]
        all_tokenized = []  # [['词1', '词2', ...], ...]

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)

            for chunk in chunks:
                chunk_id = f"{md_file.name}:{chunk['start_char']}"
                tokens = self._tokenize(chunk['text'])

                all_chunks.append({
                    'chunk_id': chunk_id,
                    'file': md_file.name,
                    'start_char': chunk['start_char'],
                    'end_char': chunk['end_char'],
                    'text_preview': chunk['text'][:100],
                })
                all_tokenized.append(tokens)

        # 保存语料库（分词后的 tokens + 元数据）
        corpus_path = kb_path / '_bm25_corpus.json'
        corpus_data = {
            'version': 'v1',
            'chunk_count': len(all_chunks),
            'chunks': all_chunks,
            'tokenized': all_tokenized,
        }

        atomic_write_json(corpus_path, corpus_data)
        self._cache.pop(str(kb_path), None)

        print(f"[BM25] 索引已构建: {len(all_chunks)} 个文本块 -> {corpus_path}")

    # ─── 查询时方法 ───

    def search(
        self, kb_path: Path, query: str, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        在指定知识库中搜索。

        Args:
            kb_path: 知识库目录路径
            query: 查询字符串
            top_k: 返回前 K 个结果

        Returns:
            [(chunk_id, bm25_score), ...] 按分数降序排列
        """
        kb_key = str(kb_path)

        # 加载或获取缓存的 BM25 模型
        signature = file_signature(Path(kb_path) / "_bm25_corpus.json")
        if kb_key not in self._cache or self._cache[kb_key].get("signature") != signature:
            self._cache.pop(kb_key, None)
            self._load_index(kb_path)

        cache = self._cache.get(kb_key)
        if cache is None:
            return []

        bm25_model = cache['bm25']
        chunks = cache['chunks']

        # 分词 + 同义词扩展
        query_tokens = self._tokenize(query)
        expanded_tokens = self._expand_query(query_tokens)

        # BM25 评分
        scores = bm25_model.get_scores(expanded_tokens)

        # 排序取 top_k
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in scored_indices:
            if score > 0:
                results.append((chunks[idx]['chunk_id'], float(score)))

        return results

    def get_chunk_text(self, kb_path: Path, chunk_id: str) -> Optional[str]:
        """
        根据 chunk_id 获取对应的文本内容。

        Args:
            kb_path: 知识库目录路径
            chunk_id: 文本块 ID（格式: "filename.md:start_char"）

        Returns:
            文本块内容，未找到时返回 None
        """
        parts = chunk_id.split(':')
        if len(parts) != 2:
            return None

        filename = parts[0]
        start_char = int(parts[1])

        file_path = Path(kb_path) / filename
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding='utf-8')
        # 返回从 start_char 开始的约 512 字符
        end_char = min(start_char + 512, len(content))
        return content[start_char:end_char]

    # ─── 内部方法 ───

    def _load_index(self, kb_path: Path) -> None:
        """从保存的语料库文件加载 BM25 索引"""
        if not HAS_BM25:
            return

        corpus_path = Path(kb_path) / '_bm25_corpus.json'
        if not corpus_path.exists():
            print(f"[BM25] 索引文件不存在: {corpus_path}，请先运行 build_index()")
            return

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        tokenized = corpus_data.get('tokenized', [])
        chunks = corpus_data.get('chunks', [])

        if not tokenized:
            return

        bm25 = BM25Okapi(tokenized)

        self._cache[str(kb_path)] = {
            'bm25': bm25,
            'chunks': chunks,
            'tokenized': tokenized,
            'signature': file_signature(corpus_path),
        }

        print(f"[BM25] 索引已加载: {len(chunks)} 个文本块")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词 + 停用词过滤。

        使用 jieba 精确模式分词，过滤停用词和单字符（除非是法律关键字）。
        """
        if not HAS_JIEBA:
            # 降级：简单按空格和标点分割
            return [w for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text) if w]

        words = jieba.cut(text, cut_all=False)
        tokens = []

        for word in words:
            word = word.strip()
            if not word:
                continue
            if word in STOPWORDS:
                continue
            # 保留：长度 >= 2 的中文词，或任意英文/数字
            if len(word) >= 2 or re.match(r'[a-zA-Z\d]+', word):
                tokens.append(word)

        return tokens

    def _expand_query(self, tokens: List[str]) -> List[str]:
        """
        使用同义词字典扩展查询词。

        例如："善意取得" -> ["善意取得", "善意受让", "即时取得"]
        """
        if not self._synonym_map:
            return tokens

        expanded = list(tokens)
        seen = set(tokens)

        for token in tokens:
            synonyms = self._synonym_map.get(token, [])
            for syn in synonyms:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)

        return expanded

    def _load_custom_dict(self) -> None:
        """加载 jieba 自定义法律词典"""
        if not HAS_JIEBA or self._custom_dict_loaded:
            return

        # 如果指定了自定义词典路径
        if self._custom_dict_path and Path(self._custom_dict_path).exists():
            jieba.load_userdict(self._custom_dict_path)
            self._custom_dict_loaded = True
            return

        # 从同义词表提取法律术语，添加为 jieba 自定义词
        if self._synonym_map:
            for term in self._synonym_map:
                if len(term) >= 2:
                    jieba.suggest_freq(term, tune=True)

        self._custom_dict_loaded = True

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        将文本按段落边界分块。

        优先在段落边界（双换行）处切分，若段落过长则在句子边界切分。

        Args:
            text: 输入文本
            chunk_size: 目标块大小（字符数）
            overlap: 块间重叠字符数

        Returns:
            [{'text': str, 'start_char': int, 'end_char': int}, ...]
        """
        if len(text) <= chunk_size:
            return [{'text': text, 'start_char': 0, 'end_char': len(text)}]

        chunks = []
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        current_text = ''
        current_start = 0
        char_pos = 0

        for para in paragraphs:
            para_with_sep = para + '\n\n'

            if len(current_text) + len(para_with_sep) > chunk_size and current_text:
                # 当前块已满，保存
                chunks.append({
                    'text': current_text.strip(),
                    'start_char': current_start,
                    'end_char': current_start + len(current_text),
                })
                # 重叠：保留末尾 overlap 字符
                if overlap > 0 and len(current_text) > overlap:
                    overlap_text = current_text[-overlap:]
                    current_start = current_start + len(current_text) - overlap
                    current_text = overlap_text + para_with_sep
                else:
                    current_start = char_pos
                    current_text = para_with_sep
            else:
                if not current_text:
                    current_start = char_pos
                current_text += para_with_sep

            char_pos += len(para_with_sep)

        # 最后一块
        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'start_char': current_start,
                'end_char': current_start + len(current_text),
            })

        return chunks


# ─── CLI 入口 ───

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python bm25_searcher.py build <kb_path>           # 构建索引")
        print("  python bm25_searcher.py search <kb_path> <query>  # 搜索")
        print("  python bm25_searcher.py --test                    # 自测")
        sys.exit(0)

    if sys.argv[1] == '--test':
        searcher = BM25Searcher()

        # 测试分块
        test_text = "第一段内容。\n\n第二段内容。\n\n第三段内容。" * 20
        chunks = BM25Searcher._chunk_text(test_text, chunk_size=100, overlap=20)
        print(f"Chunking test: {len(test_text)} chars -> {len(chunks)} chunks")

        # 测试分词
        tokens = searcher._tokenize("善意取得的构成要件包括哪些")
        print(f"Tokenize test: {tokens}")

        # 测试同义词扩展
        expanded = searcher._expand_query(tokens)
        print(f"Expansion test: {expanded}")

        print("All tests completed.")
        sys.exit(0)

    command = sys.argv[1]
    if command == 'build' and len(sys.argv) >= 3:
        searcher = BM25Searcher()
        searcher.build_index(Path(sys.argv[2]))
    elif command == 'search' and len(sys.argv) >= 4:
        searcher = BM25Searcher()
        results = searcher.search(Path(sys.argv[2]), sys.argv[3])
        for chunk_id, score in results[:10]:
            print(f"  {score:.4f}  {chunk_id}")
    else:
        print("未知命令或参数不足")
