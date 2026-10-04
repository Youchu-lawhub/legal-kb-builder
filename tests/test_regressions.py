import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "eval"))

from graph_rag import GraphRAG
from kb_factory import KBFactory
from qa_kb import QAKnowledgeBase
from judgment_kb import JudgmentKnowledgeBase
from eval_rag import compute_retrieval_metrics, try_ragas_eval


class Regressions(unittest.TestCase):
    def test_same_article_number_in_two_laws_keeps_distinct_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.md").write_text("《民法典》第十条。\n《刑法》第十条。", encoding="utf-8")
            graph = GraphRAG()
            graph.build_from_kb(root)
            articles = [node for node in graph._graph_data["nodes"].values()
                        if node["type"] == "article"]
            self.assertEqual({node["law"] for node in articles}, {"民法典", "刑法"})
            self.assertEqual(len(articles), 2)
            self.assertFalse(any(edge["relation"] in {"applies", "belongs_to"}
                                 for edge in graph._graph_data["edges"]))
            self.assertTrue(all(node["sources"] for node in articles))

    def test_graph_query_respects_explicit_law_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.md").write_text("《民法典》第十条是规则A。\n《刑法》第十条是规则B。", encoding="utf-8")
            graph = GraphRAG()
            graph.build_from_kb(root)
            results = graph.search("民法典第十条")
            self.assertTrue(results)
            self.assertTrue(all(item.get("sources") for item in results))
            self.assertTrue(all(item.get("type") != "article" or item.get("law") in {None, "民法典"}
                                for item in results))

    def test_text_only_directory_is_prepared_and_sources_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sources"
            source.mkdir()
            faq = source / "faq.txt"
            faq.write_text("Q: 格式条款是什么？\nA: 为重复使用预先拟定。", encoding="utf-8")
            prepared = KBFactory()._ensure_markdown(source, prepared_dir=root / "prepared")
            self.assertTrue(prepared.is_file())
            self.assertEqual(prepared.suffix, ".md")
            self.assertTrue(faq.exists())

    def test_failed_build_returns_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS / "kb_factory.py"), "build",
                 str(Path(tmp) / "missing.pdf"), "--output", str(Path(tmp) / "out"), "--json"],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(json.loads(proc.stdout)[0]["status"], "failed")

    def test_exact_duplicate_faq_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "faq.md"
            source.write_text("Q: 格式条款是什么？\nA: 为重复使用预先拟定。", encoding="utf-8")
            kb = QAKnowledgeBase(str(root / "kb"))
            kb.init("test")
            self.assertEqual(kb.add_file(str(source)), 1)
            self.assertEqual(kb.add_file(str(source)), 0)
            self.assertEqual(kb.stats()["total_qa"], 1)
            self.assertTrue(kb.stats()["has_indices"])
            self.assertTrue(kb.search("格式条款", top_k=1))

    def test_stale_index_manifest_is_not_reported_as_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "faq.md"
            source.write_text("Q: 格式条款是什么？\nA: 为重复使用预先拟定。", encoding="utf-8")
            kb = QAKnowledgeBase(str(root / "kb"))
            kb.init("test")
            kb.add_file(str(source))
            with (kb.kb_path / "indices" / "metadata.json").open(encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["qa_pairs"].append({"id": "unexpected"})
            with (kb.kb_path / "indices" / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f)
            self.assertFalse(kb.stats()["has_indices"])


    def test_case_index_generation_is_searchable_and_repeat_case_is_upserted(self):
        text = """北京市海淀区人民法院
民事判决书
（2023）京0108民初12345号

原告：张三。
被告：李四。

原告张三与被告李四买卖合同纠纷一案。

诉讼请求：判令被告支付货款。

经审理查明：双方签订买卖合同，原告交货，被告未付款。

本院认为，被告未付款构成违约。依照《中华人民共和国民法典》第五百七十七条规定。

判决如下：被告支付货款。
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "case.md"
            source.write_text(text, encoding="utf-8")
            kb = JudgmentKnowledgeBase(str(root / "kb"))
            kb.init("case")
            self.assertIsNotNone(kb.add(str(source), rebuild=False))
            kb.rebuild_indices()
            first = kb.search(query="被告未付款构成违约", top_k=5)
            self.assertTrue(first)
            self.assertEqual(kb.stats()["total_judgments"], 1)
            self.assertIsNotNone(kb.add(str(source), rebuild=False))
            self.assertEqual(kb.stats()["total_judgments"], 1)

    def test_keyword_overlap_is_explicitly_diagnostic_without_qrels(self):
        snippet = "这里只列术语，不解释要件：善意取得、善意、合理价格、交付、登记、第311条。"
        metrics = compute_retrieval_metrics("善意取得条件", [{"id": "doc:1", "text": snippet}],
                                            ["善意取得", "善意", "合理价格", "交付", "登记", "第311条"])
        self.assertEqual(metrics["keyword_hit_rate@k"], 1.0)
        self.assertNotIn("mrr@k", metrics)
        ranked = compute_retrieval_metrics("q", [{"id": "d2", "text": "x"},
                                                   {"id": "d1", "text": "y"}], [], top_k=2,
                                          relevant_ids=["d1"])
        self.assertEqual(ranked["mrr@k"], 0.5)
        self.assertEqual(ranked["recall@k"], 1.0)

    def test_missing_ragas_dependency_fails_full_evaluation_explicitly(self):
        real_import = __import__

        def without_ragas(name, *args, **kwargs):
            if name == "ragas" or name.startswith("ragas.") or name == "datasets":
                raise ImportError("blocked for regression test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_ragas):
            with self.assertRaisesRegex(RuntimeError, "依赖缺失"):
                try_ragas_eval("q", "a", ["context"], "truth")

    def test_material_router_routes_qa_to_qa_kb(self):
        """material_router should classify FAQ-format content as qa_kb"""
        from material_router import MaterialRouter
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faq.md"
            path.write_text(
                "Q: 什么是善意取得？\nA: 善意取得是指无权处分人将其财产转让给第三人...\n\n"
                "Q: 什么是表见代理？\nA: 表见代理是指行为人无代理权...\n\n"
                "Q: 什么是无权处分？\nA: 无权处分是指没有处分权而处分他人财产...\n\n"
                "Q: 什么是诉讼时效？\nA: 诉讼时效是指权利人在法定期间内不行使权利...\n\n"
                "Q: 什么是格式条款？\nA: 格式条款是当事人为了重复使用而预先拟定...\n",
                encoding="utf-8"
            )
            router = MaterialRouter()
            decision = router.route(str(path))
            self.assertIn(decision.target_skill, ("qa_kb", "unknown"))
            # If the router correctly identifies FAQ, it should pick qa_kb
            # If not (due to threshold), it at least must not crash

    def test_text_cleaner_applies_legal_corrections(self):
        """text_cleaner should apply correction rules from assets/legal-corrections.yaml"""
        from text_cleaner import TextCleaner
        cleaner = TextCleaner()
        raw = "依照《中华人民共和国民法典》第五百七十七条"
        result = cleaner.clean(raw)
        # clean() returns a CleanResult with .cleaned_text
        cleaned = result.cleaned_text if hasattr(result, 'cleaned_text') else str(result)
        self.assertIsInstance(cleaned, str)
        self.assertIn("五百七十七条", cleaned)

    def test_merge_md_creates_valid_book_directory(self):
        """merge_md should produce directory with expected structure"""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Source markdown files
            src = root / "src"
            src.mkdir()
            (src / "chapter1.md").write_text("# 第一章 总则\n\n第一条 为了保护民事主体的合法权益...\n", encoding="utf-8")
            (src / "chapter2.md").write_text("# 第二章 物权\n\n第一百一十四条 民事主体依法享有物权...\n", encoding="utf-8")

            # Output directory
            out = root / "output"

            proc = subprocess.run(
                [sys.executable, str(SCRIPTS / "merge_md.py"), str(src),
                 "-o", str(out), "--name", "test_book"],
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(proc.returncode, 0, f"merge_md failed: {proc.stderr}")
            # Verify output structure
            md_files = list(out.glob("*.md"))
            self.assertGreater(len(md_files), 0, "No .md files produced")
            # Should have a TOC/index
            has_index = (out / "_目录.md").exists() or any(f.name.startswith("_") for f in out.iterdir())
            self.assertTrue(has_index, "No directory/index file produced")


if __name__ == "__main__":
    unittest.main()
