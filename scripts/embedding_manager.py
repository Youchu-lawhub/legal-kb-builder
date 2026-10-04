#!/usr/bin/env python3
"""
向量嵌入与 FAISS 检索管理模块（v1.2 - 多模型可配置 + 父子分块版）

支持切换多种中文嵌入模型：
  - BAAI/bge-small-zh-v1.5  (384维, 默认, 轻量快速)
  - BAAI/bge-large-zh-v1.5  (1024维, 精度更高)
  - BAAI/bge-m3             (1024维, 多语言+多粒度, 2025 前沿)

支持两种分块策略：
  - flat（默认）: 段落边界分块，单一粒度
  - parent_child: 父子分块（Small-to-Big）
      检索阶段用小块（child, ~256字符）保证精度
      生成阶段用父块（parent, ~1024字符）保证上下文完整性
      _chunks.json 增加 parent_id 字段关联父子块

通过 SentenceTransformer 生成嵌入，FAISS 索引提供高效向量相似度检索。
模型切换时自动检测维度变化，提示重建索引。

使用方式：
    from embedding_manager import EmbeddingManager

    # 默认模型 + 默认 flat 分块
    em = EmbeddingManager()

    # 切换高级模型
    em = EmbeddingManager(model_name='BAAI/bge-m3')

    # 启用父子分块（推荐生产环境）
    em.build_index(kb_path, chunk_strategy='parent_child')

    # 查询
    results = em.search(kb_path, "善意取得的构成要件", top_k=20)
    # -> [(chunk_id, cosine_score), ...]

依赖：
    pip install sentence-transformers faiss-cpu torch numpy
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import shutil

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ─── 常量 ───
DEFAULT_MODEL = 'BAAI/bge-small-zh-v1.5'
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'legal-kb-builder' / 'models'

# 模型注册表：记录各模型的输出维度（用于维度校验和索引重建判断）
# None 表示需运行时动态获取（如 bge-m3 的 ColBERT 模式，但 Dense 模式仍为 1024）
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'BAAI/bge-small-zh-v1.5': {
        'dim': 384,
        'desc': '轻量级，推理快，适合低配环境',
        'normalize': True,
    },
    'BAAI/bge-large-zh-v1.5': {
        'dim': 1024,
        'desc': '精度更高，MTEB 中文榜领先 small',
        'normalize': True,
    },
    'BAAI/bge-m3': {
        'dim': 1024,
        'desc': '多语言+多粒度(可支持Dense/Sparse/ColBERT)，2025 前沿',
        'normalize': True,
    },
}


def get_model_dim(model_name: str) -> Optional[int]:
    """获取模型输出维度（来自注册表）"""
    info = MODEL_REGISTRY.get(model_name)
    return info['dim'] if info else None


class EmbeddingManager:
    """
    向量嵌入管理器（多模型可配置）

    Args:
        model_name: SentenceTransformer 模型名称
                    支持 bge-small-zh-v1.5 / bge-large-zh-v1.5 / bge-m3
        cache_dir: 模型缓存目录
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str = None,
    ):
        self._model_name = model_name
        self._cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self._model: Optional[Any] = None  # 延迟加载
        self._dim: Optional[int] = get_model_dim(model_name)  # 从注册表预取维度

        # 运行时缓存：{kb_path_str: {index, chunks}}
        self._index_cache: Dict[str, Any] = {}

    # ─── 构建时方法 ───

    def build_index(
        self,
        kb_path: Path,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 64,
        chunk_strategy: str = "flat",
        child_size: int = 256,
        parent_size: int = 1024,
    ) -> None:
        """
        为指定知识库构建 FAISS 向量索引。

        读取所有非索引 .md 文件，分块，生成嵌入，建立 FAISS 索引。

        Args:
            kb_path: 知识库目录路径
            chunk_size: 每块字符数（flat 策略用，默认 512）
            chunk_overlap: 块间重叠字符数（flat 策略用，默认 50）
            batch_size: 嵌入生成的批大小
            chunk_strategy: 分块策略
                - "flat"（默认）: 段落边界分块，单一粒度
                - "parent_child": 父子分块（小块检索+大块上下文）
            child_size: 子块字符数（parent_child 策略用，默认 256）
            parent_size: 父块字符数（parent_child 策略用，默认 1024）
        """
        kb_path = Path(kb_path)
        from shared_utils import file_lock
        generation = uuid.uuid4().hex
        with file_lock(kb_path / ".vector-write.lock", timeout=3600):
            try:
                return self._build_index_locked(kb_path, chunk_size, chunk_overlap, batch_size, chunk_strategy, child_size, parent_size, generation)
            except Exception:
                shutil.rmtree(kb_path / "_vector_generations" / (".tmp-" + generation), ignore_errors=True)
                raise

    def _build_index_locked(self, kb_path, chunk_size, chunk_overlap, batch_size, chunk_strategy, child_size, parent_size, generation):
        self._ensure_dependencies()
        self._ensure_model_loaded()
        generations = kb_path / "_vector_generations"
        generations.mkdir(parents=True, exist_ok=True)
        staging = generations / (".tmp-" + generation)
        staging.mkdir()
        md_files = sorted(
            f for f in kb_path.glob("*.md")
            if not f.name.startswith("_")
        )

        if not md_files:
            raise ValueError(f"[Embedding] 未在 {kb_path} 找到 MD 文件")

        # 分块
        all_chunks = []
        all_texts = []
        parent_chunks = {}  # parent_id -> parent_text（仅 parent_child 策略）

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            if chunk_strategy == "parent_child":
                chunks = self._chunk_parent_child(
                    content, child_size, parent_size, md_file.name
                )
            else:
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                # 补充 chunk_id 字段（flat 策略）
                for ch in chunks:
                    ch['chunk_id'] = f"{md_file.name}:{ch['start_char']}"
                    ch['parent_id'] = None  # flat 策略无父块

            for chunk in chunks:
                all_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'file': md_file.name,
                    'start_char': chunk['start_char'],
                    'end_char': chunk['end_char'],
                    'text_preview': chunk['text'][:100],
                    'parent_id': chunk.get('parent_id'),
                    'chunk_type': chunk.get('chunk_type', 'flat'),
                })
                all_texts.append(chunk['text'])

                # 收集父块文本（parent_child 策略）
                if chunk.get('parent_id') and chunk['parent_id'] not in parent_chunks:
                    parent_chunks[chunk['parent_id']] = chunk.get('parent_text', '')

        print(f"[Embedding] 策略={chunk_strategy}, 共 {len(all_texts)} 个文本块，开始生成嵌入...")

        # 生成嵌入（批处理）—— 只对子块/flat块生成嵌入（父块不单独嵌入）
        embeddings = self._model.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=len(all_texts) > 100,
            normalize_embeddings=True,  # L2 归一化，使内积 = 余弦相似度
        )

        # 构建 FAISS 索引（使用 Inner Product，因为已归一化）
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        # 保存 FAISS 索引
        faiss_path = staging / '_vectors.faiss'
        faiss.write_index(index, str(faiss_path))

        # 保存块元数据
        chunks_path = staging / '_chunks.json'
        chunks_data = {
            'version': 'v1.2',
            'model': self._model_name,
            'dim': dim,
            'chunk_strategy': chunk_strategy,
            'chunk_count': len(all_chunks),
            'chunks': all_chunks,
        }
        # parent_child 策略额外保存父块文本映射
        if chunk_strategy == "parent_child" and parent_chunks:
            chunks_data['parent_chunks'] = parent_chunks
            chunks_data['parent_count'] = len(parent_chunks)

        from shared_utils import atomic_write_json, file_signature
        atomic_write_json(chunks_path, chunks_data)
        destination = generations / generation
        os.replace(staging, destination)
        atomic_write_json(kb_path / "_vector_manifest.json", {"schema_version": 1, "generation": generation})
        self._index_cache.pop(str(kb_path), None)
        return {"generation": generation, "chunk_count": len(all_chunks)}

        print(f"[Embedding] 索引已构建: {len(all_chunks)} 个块, {dim} 维, 策略={chunk_strategy}")
        if parent_chunks:
            print(f"  父块数: {len(parent_chunks)}")
        print(f"  FAISS: {faiss_path}")
        print(f"  Metadata: {chunks_path}")

    # ─── 查询时方法 ───

    def search(
        self, kb_path: Path, query: str, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        在指定知识库中执行向量相似度搜索。

        Args:
            kb_path: 知识库目录路径
            query: 查询字符串
            top_k: 返回前 K 个结果

        Returns:
            [(chunk_id, cosine_score), ...] 按相似度降序排列
        """
        self._ensure_dependencies()
        self._ensure_model_loaded()

        kb_key = str(kb_path)

        # 加载或获取缓存的 FAISS 索引
        if kb_key not in self._index_cache:
            self._load_index(kb_path)

        cache = self._index_cache.get(kb_key)
        if cache is None:
            return []

        index = cache['index']
        chunks = cache['chunks']

        # 生成查询嵌入
        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS 搜索
        if top_k < 1 or index.ntotal == 0:
            return []
        actual_k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                results.append((chunks[idx]['chunk_id'], float(score)))

        return results

    def get_chunk_text(self, kb_path: Path, chunk_id: str) -> Optional[str]:
        """
        根据 chunk_id 获取对应的文本内容。

        对于 parent_child 策略，若该 chunk 有 parent_id，
        则返回父块文本（完整上下文），而非子块文本。

        Args:
            kb_path: 知识库目录路径
            chunk_id: 文本块 ID（格式: "filename.md:start_char"）

        Returns:
            文本块内容（parent_child 策略返回父块），未找到时返回 None
        """
        kb_path = Path(kb_path)
        from shared_utils import vector_paths
        _, chunks_path = vector_paths(kb_path)

        if not chunks_path.exists():
            return None

        # 加载 chunks 元数据（带缓存）
        kb_key = str(kb_path)
        if kb_key not in self._index_cache or self._index_cache[kb_key] is None:
            self._load_index(kb_path)

        cached = self._index_cache.get(kb_key)
        if not cached:
            # 降级：无法加载元数据，用旧逻辑直接读文件
            return self._read_chunk_from_file(kb_path, chunk_id)

        chunks = cached.get('chunks', [])
        parent_chunks = cached.get('parent_chunks', {})

        # 查找目标 chunk
        target = None
        for ch in chunks:
            if ch['chunk_id'] == chunk_id:
                target = ch
                break

        if not target:
            return self._read_chunk_from_file(kb_path, chunk_id)

        # parent_child 策略：优先返回父块文本
        parent_id = target.get('parent_id')
        if parent_id and parent_id in parent_chunks:
            return parent_chunks[parent_id]

        # flat 策略或无父块映射：从文件读取
        return self._read_chunk_from_file(kb_path, chunk_id, target)

    def get_parent_text(self, kb_path: Path, chunk_id: str) -> Optional[str]:
        """
        显式获取父块文本（parent_child 策略专用）。

        若为 flat 策略或无父块，返回 None。
        """
        kb_path = Path(kb_path)
        from shared_utils import vector_paths
        _, chunks_path = vector_paths(kb_path)

        if not chunks_path.exists():
            return None

        kb_key = str(kb_path)
        if kb_key not in self._index_cache or self._index_cache[kb_key] is None:
            self._load_index(kb_path)

        cached = self._index_cache.get(kb_key)
        if not cached:
            return None

        chunks = cached.get('chunks', [])
        parent_chunks = cached.get('parent_chunks', {})

        for ch in chunks:
            if ch['chunk_id'] == chunk_id:
                parent_id = ch.get('parent_id')
                if parent_id and parent_id in parent_chunks:
                    return parent_chunks[parent_id]
                return None

        return None

    def _read_chunk_from_file(
        self, kb_path: Path, chunk_id: str, meta: Dict = None
    ) -> Optional[str]:
        """从原始文件读取 chunk 文本（降级路径）"""
        parts = chunk_id.split(':')
        if len(parts) < 2:
            return None

        filename = parts[0]
        try:
            start_char = int(parts[1])
        except ValueError:
            return None

        file_path = Path(kb_path) / filename
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding='utf-8')
        if meta and 'end_char' in meta:
            end_char = min(meta['end_char'], len(content))
        else:
            end_char = min(start_char + 512, len(content))
        return content[start_char:end_char]

    def embed_texts(self, texts: List[str]) -> Any:
        """
        为任意文本列表生成嵌入。

        供外部模块直接使用（如 case judgment skill 的字段嵌入）。

        Args:
            texts: 文本列表

        Returns:
            numpy ndarray of shape (len(texts), EMBEDDING_DIM)
        """
        self._ensure_dependencies()
        self._ensure_model_loaded()

        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        ).astype(np.float32)

    # ─── 内部方法 ───

    def _ensure_dependencies(self) -> None:
        """确保必要依赖已安装"""
        missing = []
        if not HAS_NUMPY:
            missing.append('numpy')
        if not HAS_FAISS:
            missing.append('faiss-cpu')
        if not HAS_ST:
            missing.append('sentence-transformers')
        if missing:
            raise ImportError(
                f"缺少依赖: {', '.join(missing)}。"
                f"请运行: pip install {' '.join(missing)}"
            )

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载（延迟加载）"""
        if self._model is not None:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Embedding] 加载模型: {self._model_name}")
        self._model = SentenceTransformer(
            self._model_name,
            cache_folder=str(self._cache_dir),
        )
        actual_dim = self._model.get_sentence_embedding_dimension()
        print(f"[Embedding] 模型已就绪 (维度: {actual_dim})")

        # 校验注册表维度与实际维度（发现不一致时更新缓存）
        if self._dim is None:
            self._dim = actual_dim
        elif actual_dim != self._dim:
            print(f"[Embedding] ⚠️ 注册表维度({self._dim})与实际({actual_dim})不符，以实际为准")
            self._dim = actual_dim

    def _load_index(self, kb_path: Path) -> None:
        """从文件加载 FAISS 索引和元数据（含模型一致性校验）"""
        kb_path = Path(kb_path)
        from shared_utils import vector_paths, file_signature
        faiss_path, chunks_path = vector_paths(kb_path)
        signature = (file_signature(faiss_path), file_signature(chunks_path))
        cached = self._index_cache.get(str(kb_path))
        if cached and cached.get('signature') == signature:
            return
        self._index_cache.pop(str(kb_path), None)

        if not faiss_path.exists() or not chunks_path.exists():
            print(f"[Embedding] 索引文件不存在: {kb_path}，请先运行 build_index()")
            return

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        # ─── 模型一致性校验（v1.1 新增）───
        # 防止用 A 模型建的索引被 B 模型查询（维度不一致会导致检索错误）
        stored_model = chunks_data.get('model', 'unknown')
        stored_dim = chunks_data.get('dim', 0)

        if stored_model != self._model_name:
            print(f"[Embedding] ⚠️ 模型不一致:")
            print(f"  索引构建时使用的模型: {stored_model} (dim={stored_dim})")
            print(f"  当前配置的模型: {self._model_name} (dim={self._dim})")
            print(f"  维度不一致会导致检索结果错误。请用当前模型重建索引:")
            print(f"    python embedding_manager.py build {kb_path}")
            # 仍加载旧索引但标记不匹配（允许降级使用，但不推荐）
            self._index_cache[str(kb_path)] = None
            return

        if self._dim and stored_dim != self._dim:
            print(f"[Embedding] ⚠️ 维度不匹配: 索引={stored_dim}, 当前模型={self._dim}")
            print(f"  请重建索引: python embedding_manager.py build {kb_path}")
            self._index_cache[str(kb_path)] = None
            return

        index = faiss.read_index(str(faiss_path))

        self._index_cache[str(kb_path)] = {
            'index': index,
            'chunks': chunks_data.get('chunks', []),
            'parent_chunks': chunks_data.get('parent_chunks', {}),  # v1.2 父块映射
            'chunk_strategy': chunks_data.get('chunk_strategy', 'flat'),
            'signature': signature,
        }

        strategy = chunks_data.get('chunk_strategy', 'flat')
        parent_count = len(chunks_data.get('parent_chunks', {}))
        print(f"[Embedding] 索引已加载: {index.ntotal} 个向量 (model={stored_model}, dim={stored_dim}, strategy={strategy}")
        if parent_count:
            print(f"  父块映射: {parent_count} 个")

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """
        将文本按段落边界分块。

        与 bm25_searcher.py 中的实现一致（使用相同的分块策略确保 chunk_id 对齐）。
        """
        if len(text) <= chunk_size:
            return [{'text': text, 'start_char': 0, 'end_char': len(text)}]

        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)

        current_text = ''
        current_start = 0
        char_pos = 0

        for para in paragraphs:
            para_with_sep = para + '\n\n'

            if len(current_text) + len(para_with_sep) > chunk_size and current_text:
                chunks.append({
                    'text': current_text.strip(),
                    'start_char': current_start,
                    'end_char': current_start + len(current_text),
                })
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

        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'start_char': current_start,
                'end_char': current_start + len(current_text),
            })

        return chunks

    @staticmethod
    def _chunk_parent_child(
        text: str,
        child_size: int = 256,
        parent_size: int = 1024,
        filename: str = "",
    ) -> List[Dict[str, Any]]:
        """
        父子分块（Small-to-Big）策略。

        先按 parent_size 切父块（大粒度，保证上下文完整），
        再在每个父块内按 child_size 切子块（小粒度，保证检索精度）。
        只对子块生成嵌入，检索命中子块时返回对应父块文本。

        Args:
            text: 原始文本
            child_size: 子块字符数（默认 256，检索精度优先）
            parent_size: 父块字符数（默认 1024，上下文完整优先）
            filename: 文件名（用于生成 chunk_id）

        Returns:
            [{text, start_char, end_char, chunk_id, parent_id, parent_text, chunk_type}, ...]
            每个子块都带有 parent_id 和 parent_text
        """
        if len(text) <= parent_size:
            # 文本不长，父块=整个文本，子块=父块本身
            parent_id = f"{filename}:0"
            return [{
                'text': text,
                'start_char': 0,
                'end_char': len(text),
                'chunk_id': f"{filename}:0",
                'parent_id': parent_id,
                'parent_text': text,
                'chunk_type': 'child',
            }]

        # Step 1: 按段落边界切父块（尽量在段落边界切，保证语义完整）
        parent_chunks_raw = []
        paragraphs = re.split(r'\n\s*\n', text)

        current_text = ''
        current_start = 0
        char_pos = 0

        for para in paragraphs:
            para_with_sep = para + '\n\n'

            if len(current_text) + len(para_with_sep) > parent_size and current_text:
                parent_chunks_raw.append({
                    'text': current_text.strip(),
                    'start_char': current_start,
                    'end_char': current_start + len(current_text),
                })
                current_start = char_pos
                current_text = para_with_sep
            else:
                if not current_text:
                    current_start = char_pos
                current_text += para_with_sep

            char_pos += len(para_with_sep)

        if current_text.strip():
            parent_chunks_raw.append({
                'text': current_text.strip(),
                'start_char': current_start,
                'end_char': current_start + len(current_text),
            })

        # Step 2: 在每个父块内切子块
        result = []
        for parent_idx, parent in enumerate(parent_chunks_raw):
            parent_text = parent['text']
            parent_start = parent['start_char']
            parent_id = f"{filename}:parent_{parent_idx}"

            if len(parent_text) <= child_size:
                # 父块本身够短，直接作为一个子块
                result.append({
                    'text': parent_text,
                    'start_char': parent_start,
                    'end_char': parent['end_char'],
                    'chunk_id': f"{filename}:{parent_start}",
                    'parent_id': parent_id,
                    'parent_text': parent_text,
                    'chunk_type': 'child',
                })
                continue

            # 在父块内按 child_size 切子块（带少量 overlap 保持连贯）
            child_overlap = max(20, child_size // 10)
            pos = 0
            while pos < len(parent_text):
                end = min(pos + child_size, len(parent_text))
                child_text = parent_text[pos:end]

                result.append({
                    'text': child_text,
                    'start_char': parent_start + pos,
                    'end_char': parent_start + end,
                    'chunk_id': f"{filename}:{parent_start + pos}",
                    'parent_id': parent_id,
                    'parent_text': parent_text,  # 完整父块文本
                    'chunk_type': 'child',
                })

                if end >= len(parent_text):
                    break
                pos = end - child_overlap

        return result


# ─── CLI 入口 ───

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python embedding_manager.py build <kb_path> [--model NAME] [--strategy S]  # 构建索引")
        print("  python embedding_manager.py search <kb_path> <query> [--model NAME]        # 搜索")
        print("  python embedding_manager.py --models                                        # 列出可用模型")
        print("  python embedding_manager.py --test                                          # 自测（不需要模型）")
        print("\n分块策略 (--strategy):")
        print("  flat          段落边界分块，单一粒度（默认）")
        print("  parent_child  父子分块，小块检索+大块上下文（推荐生产环境）")
        sys.exit(0)

    if sys.argv[1] == '--models':
        print("可用嵌入模型:")
        for name, info in MODEL_REGISTRY.items():
            default = " (默认)" if name == DEFAULT_MODEL else ""
            print(f"  {name}")
            print(f"    维度: {info['dim']}, {info['desc']}{default}")
        sys.exit(0)

    if sys.argv[1] == '--test':
        # 基础自测（不需要模型下载）
        chunks = EmbeddingManager._chunk_text(
            "第一段。\n\n第二段。\n\n第三段。" * 20,
            chunk_size=100, overlap=20
        )
        print(f"flat 分块测试: {len(chunks)} chunks")

        # 父子分块测试
        sample = "第一段内容。\n\n第二段内容。\n\n第三段内容。" * 10
        pc_chunks = EmbeddingManager._chunk_parent_child(
            sample, child_size=50, parent_size=150, filename="test.md"
        )
        print(f"parent_child 分块测试: {len(pc_chunks)} 子块")
        if pc_chunks:
            first = pc_chunks[0]
            print(f"  首块 parent_id: {first['parent_id']}")
            print(f"  首块 parent_text 长度: {len(first.get('parent_text', ''))}")

        # 检查依赖状态
        print(f"\nnumpy: {'yes' if HAS_NUMPY else 'no'}")
        print(f"faiss: {'yes' if HAS_FAISS else 'no'}")
        print(f"sentence-transformers: {'yes' if HAS_ST else 'no'}")
        print("\n可用模型:")
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name} (dim={info['dim']})")
        print("All basic tests completed.")
        sys.exit(0)

    command = sys.argv[1]

    # 解析 --model 参数
    model_name = DEFAULT_MODEL
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]

    # 解析 --strategy 参数
    chunk_strategy = "flat"
    if '--strategy' in sys.argv:
        idx = sys.argv.index('--strategy')
        if idx + 1 < len(sys.argv):
            chunk_strategy = sys.argv[idx + 1]
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]

    if command == 'build' and len(sys.argv) >= 3:
        em = EmbeddingManager(model_name=model_name)
        em.build_index(Path(sys.argv[2]), chunk_strategy=chunk_strategy)
    elif command == 'search' and len(sys.argv) >= 4:
        em = EmbeddingManager(model_name=model_name)
        results = em.search(Path(sys.argv[2]), sys.argv[3])
        for chunk_id, score in results[:10]:
            print(f"  {score:.4f}  {chunk_id}")
    else:
        print("未知命令或参数不足")
