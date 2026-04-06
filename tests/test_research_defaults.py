import unittest
from pathlib import Path

from src.config_utils import get_default_config
from src.research import EvaluationPipeline, default_threshold_list, resolve_research_defaults


class ResearchDefaultsTests(unittest.TestCase):
    def test_builtin_defaults_resolve_correctly(self):
        config = get_default_config()

        resolved = resolve_research_defaults(config)

        self.assertEqual(resolved.stage12.fixed_feature_set_name, "baseline_core")
        self.assertEqual(resolved.stage4.working_target_id, "return_threshold_h3_0_05pct")
        self.assertEqual(list(resolved.common.threshold_list), default_threshold_list())
        self.assertEqual(list(resolved.stage5.feature_set_names), ["baseline_core", "all_eligible"])
        self.assertEqual(list(resolved.stage5.preset_names), ["conservative", "balanced"])

    def test_resolve_research_defaults_honors_overrides(self):
        config = get_default_config()
        config["research"]["defaults"]["common"]["threshold_list"] = [0.50, 0.51, 0.52]
        config["research"]["defaults"]["stage5"]["feature_set_names"] = ["baseline_core"]
        config["research"]["defaults"]["stage5"]["preset_names"] = ["conservative"]
        config["research"]["defaults"]["truth_gate"]["minimum_test_coverage"] = 0.15

        resolved = resolve_research_defaults(config)

        self.assertEqual(resolved.common.threshold_list, [0.50, 0.51, 0.52])
        self.assertEqual(resolved.stage5.feature_set_names, ["baseline_core"])
        self.assertEqual(resolved.stage5.preset_names, ["conservative"])
        self.assertEqual(resolved.truth_gate.minimum_test_coverage, 0.15)

    def test_resolve_research_defaults_supports_legacy_stage5_alias(self):
        config = get_default_config()
        del config["research"]["defaults"]["stage5"]
        config["research"]["stage5_defaults"] = {
            "target_ids": ["return_threshold_h3_0_05pct"],
            "feature_sets": ["baseline_core"],
            "presets": ["conservative"],
        }

        resolved = resolve_research_defaults(config)

        self.assertEqual(resolved.stage5.target_ids, ["return_threshold_h3_0_05pct"])
        self.assertEqual(resolved.stage5.feature_set_names, ["baseline_core"])
        self.assertEqual(resolved.stage5.preset_names, ["conservative"])

    def test_resolve_research_defaults_rejects_invalid_override(self):
        config = get_default_config()
        config["research"]["defaults"]["stage5"]["max_worker_cap"] = 1
        config["research"]["defaults"]["stage5"]["min_auto_workers"] = 2

        with self.assertRaisesRegex(ValueError, "min_auto_workers"):
            resolve_research_defaults(config)

    def test_evaluation_pipeline_fallback_uses_shared_thresholds(self):
        pipeline = EvaluationPipeline()

        self.assertEqual(list(pipeline.thresholds), default_threshold_list())

    def test_old_threshold_literal_removed_from_research_modules(self):
        repo_root = Path(__file__).resolve().parents[1]
        banned_literal = "[0.50, 0.55, 0.60, 0.65, 0.70, 0.75]"
        guarded_files = [
            repo_root / "src" / "research" / "schemas.py",
            repo_root / "src" / "research" / "evaluation_pipeline.py",
            repo_root / "src" / "services" / "research_service.py",
        ]

        for file_path in guarded_files:
            self.assertNotIn(banned_literal, file_path.read_text(encoding="utf-8"), msg=str(file_path))


if __name__ == "__main__":
    unittest.main()
