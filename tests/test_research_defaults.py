import unittest
from pathlib import Path

from src.config_utils import get_default_config
from src.research import EvaluationPipeline, default_threshold_list, resolve_research_defaults


class ResearchDefaultsTests(unittest.TestCase):
    def test_builtin_defaults_resolve_correctly(self):
        config = get_default_config()

        resolved = resolve_research_defaults(config)

        self.assertEqual(resolved.search.working_target_id, "return_threshold_h3_0_05pct")
        self.assertEqual(list(resolved.common.threshold_list), default_threshold_list())
        self.assertEqual(list(resolved.search.feature_set_names), ["baseline_core", "all_eligible"])
        self.assertEqual(resolved.search.trainer_name, "current_ensemble")
        self.assertEqual(list(resolved.search.preset_names), ["conservative", "balanced"])

    def test_resolve_research_defaults_honors_overrides(self):
        config = get_default_config()
        config["research"]["defaults"]["common"]["threshold_list"] = [0.50, 0.51, 0.52]
        config["research"]["defaults"]["search"]["feature_set_names"] = ["baseline_core"]
        config["research"]["defaults"]["search"]["trainer_name"] = "lstm"
        config["research"]["defaults"]["search"]["preset_names"] = ["conservative"]
        config["research"]["defaults"]["truth_gate"]["minimum_test_coverage"] = 0.15

        resolved = resolve_research_defaults(config)

        self.assertEqual(resolved.common.threshold_list, [0.50, 0.51, 0.52])
        self.assertEqual(resolved.search.feature_set_names, ["baseline_core"])
        self.assertEqual(resolved.search.trainer_name, "lstm")
        self.assertEqual(resolved.search.preset_names, ["conservative"])
        self.assertEqual(resolved.truth_gate.minimum_test_coverage, 0.15)

    def test_resolve_research_defaults_rejects_invalid_worker_range(self):
        config = get_default_config()
        config["research"]["defaults"]["search"]["max_worker_cap"] = 1
        config["research"]["defaults"]["search"]["min_auto_workers"] = 2

        with self.assertRaisesRegex(ValueError, "min_auto_workers"):
            resolve_research_defaults(config)

    def test_resolve_research_defaults_rejects_invalid_search_trainer_type(self):
        config = get_default_config()
        config["research"]["defaults"]["search"]["trainer_name"] = []

        with self.assertRaisesRegex(ValueError, "search.trainer_name"):
            resolve_research_defaults(config)

    def test_evaluation_pipeline_fallback_uses_shared_thresholds(self):
        pipeline = EvaluationPipeline()

        self.assertEqual(list(pipeline.thresholds), default_threshold_list())

    def test_old_threshold_literal_removed_from_research_modules(self):
        repo_root = Path(__file__).resolve().parents[1]
        banned_literal = "[0.50, 0.55, 0.60, 0.65, 0.70, 0.75]"
        guarded_files = [
            repo_root / "src" / "research" / "schemas.py",
            repo_root / "src" / "research" / "execution" / "evaluation_pipeline.py",
            repo_root / "src" / "services" / "research_service.py",
        ]

        for file_path in guarded_files:
            self.assertNotIn(banned_literal, file_path.read_text(encoding="utf-8"), msg=str(file_path))


if __name__ == "__main__":
    unittest.main()
