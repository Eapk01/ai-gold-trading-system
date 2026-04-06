"""Stage 5 bounded search orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .schemas import SearchCandidateSummary, SearchRequest
from .search_truth_gate import apply_truth_gate, build_truth_gate_summary


@dataclass(frozen=True)
class SearchCandidateConfig:
    """One bounded Stage 5 candidate configuration."""

    candidate_id: str
    target_spec: Dict[str, Any]
    target_spec_id: str
    target_display_name: str
    feature_set_name: str
    preset_name: str
    trainer_name: str
    trainer_params: Dict[str, Any]


@dataclass
class SearchRunner:
    """Build and rank bounded Stage 5 search candidates."""

    def build_candidate_grid(
        self,
        *,
        request: SearchRequest,
        preset_definitions: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> List[SearchCandidateConfig]:
        candidates: List[SearchCandidateConfig] = []
        for target_spec in request.resolved_target_specs():
            target_spec_id = str(target_spec.get("spec_id") or "target")
            target_display_name = str(target_spec.get("display_name") or target_spec_id)
            for feature_set_name in request.feature_set_names:
                for preset_name in request.preset_names:
                    candidates.append(
                        SearchCandidateConfig(
                            candidate_id=f"{request.search_id}_{target_spec_id}_{feature_set_name}_{preset_name}",
                            target_spec=dict(target_spec),
                            target_spec_id=target_spec_id,
                            target_display_name=target_display_name,
                            feature_set_name=feature_set_name,
                            preset_name=preset_name,
                            trainer_name=request.trainer_name,
                            trainer_params=dict(preset_definitions[preset_name]),
                        )
                    )
        return candidates

    def rank_candidates(
        self,
        candidates: Iterable[SearchCandidateSummary],
        truth_gate_defaults: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raw_candidates = list(candidates)
        completed_candidates = [
            apply_truth_gate(candidate, truth_gate_defaults=truth_gate_defaults)
            for candidate in raw_candidates
            if str(candidate.execution_status or "completed") == "completed"
        ]
        failed_candidates = sorted(
            [
                candidate
                for candidate in raw_candidates
                if str(candidate.execution_status or "completed") != "completed"
            ],
            key=lambda candidate: (
                str(candidate.target_spec_id or ""),
                str(candidate.feature_set_name or ""),
                str(candidate.preset_name or ""),
                str(candidate.candidate_id or ""),
            ),
        )
        ordered_completed_candidates = sorted(completed_candidates, key=self._ranking_key)
        ordered_candidates = ordered_completed_candidates + failed_candidates
        leaderboard_rows: List[Dict[str, Any]] = []
        gate_summary = build_truth_gate_summary(
            ordered_completed_candidates,
            truth_gate_defaults=truth_gate_defaults,
        )
        recommended_winner: Dict[str, Any] = {
            "status": "no_winner",
            "reason": "No candidate passed the full Stage 5.1 truth gate.",
            "gate_summary": gate_summary,
        }

        passing_candidates = [candidate for candidate in ordered_completed_candidates if candidate.passed_truth_gate]
        recommended_candidate = passing_candidates[0] if passing_candidates else None
        for rank, candidate in enumerate(ordered_candidates, start=1):
            execution_status = str(candidate.execution_status or "completed")
            integrity = dict((candidate.diagnostics or {}).get("integrity") or {})
            is_recommended = recommended_candidate is not None and candidate.candidate_id == recommended_candidate.candidate_id
            candidate.is_recommended = is_recommended
            candidate.recommendation_reason = (
                "Recommended winner after validation ranking and full truth gate."
                if is_recommended
                else (
                    f"Candidate execution failed: {candidate.error_message or 'Unknown error'}"
                    if execution_status != "completed"
                    else (
                        "Failed truth gate: " + ", ".join(candidate.truth_gate_failures)
                        if candidate.truth_gate_failures
                        else ""
                    )
                )
            )
            leaderboard_rows.append(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "experiment_id": candidate.experiment_id,
                    "experiment_name": candidate.experiment_name,
                    "target_spec_id": candidate.target_spec_id,
                    "target_display_name": candidate.target_display_name,
                    "feature_set_name": candidate.feature_set_name,
                    "preset_name": candidate.preset_name,
                    "selected_threshold": candidate.selected_threshold,
                    "validation_beat_rate": candidate.validation_summary.get("beat_rate"),
                    "validation_f1_std": candidate.validation_summary.get("f1_std"),
                    "validation_mean_f1": candidate.validation_summary.get("mean_f1"),
                    "validation_mean_coverage": candidate.validation_summary.get("mean_coverage"),
                    "test_mean_f1": candidate.test_summary.get("mean_f1"),
                    "test_mean_coverage": candidate.test_summary.get("mean_coverage"),
                    "overall_mean_test_accuracy": candidate.overall_mean_test_accuracy,
                    "majority_baseline_mean_test_accuracy": candidate.majority_baseline_mean_test_accuracy,
                    "test_best_baseline_mean_f1": candidate.test_summary.get("best_baseline_mean_f1"),
                    "passed_test_guardrail": candidate.passed_test_guardrail,
                    "passed_truth_gate": candidate.passed_truth_gate,
                    "truth_gate_failures": list(candidate.truth_gate_failures),
                    "proof_status": integrity.get("proof_status"),
                    "integrity_contract_ok": bool(integrity.get("integrity_contract_ok")),
                    "execution_status": execution_status,
                    "error_message": candidate.error_message,
                    "elapsed_seconds": candidate.elapsed_seconds,
                    "diagnostics": dict(candidate.diagnostics),
                    "report_file": candidate.report_file,
                    "is_recommended": is_recommended,
                }
            )

        if recommended_candidate is not None:
            recommended_integrity = dict((recommended_candidate.diagnostics or {}).get("integrity") or {})
            recommended_winner = {
                "status": "recommended",
                "reason": "Recommended winner after validation ranking and full truth gate.",
                "gate_summary": gate_summary,
                "candidate_id": recommended_candidate.candidate_id,
                "experiment_id": recommended_candidate.experiment_id,
                "experiment_name": recommended_candidate.experiment_name,
                "report_file": recommended_candidate.report_file,
                "target_spec_id": recommended_candidate.target_spec_id,
                "target_display_name": recommended_candidate.target_display_name,
                "feature_set_name": recommended_candidate.feature_set_name,
                "preset_name": recommended_candidate.preset_name,
                "selected_threshold": recommended_candidate.selected_threshold,
                "truth_gate_failures": list(recommended_candidate.truth_gate_failures),
                "proof_status": recommended_integrity.get("proof_status"),
                "integrity_contract_ok": bool(recommended_integrity.get("integrity_contract_ok")),
                "diagnostics": dict(recommended_candidate.diagnostics),
            }

        return {
            "ordered_candidates": ordered_candidates,
            "leaderboard_rows": leaderboard_rows,
            "recommended_winner": recommended_winner,
            "gate_summary": gate_summary,
        }

    def _ranking_key(self, candidate: SearchCandidateSummary) -> tuple[float, float, float, float, float]:
        validation_summary = candidate.validation_summary or {}
        return (
            -float(validation_summary.get("beat_rate") or 0.0),
            float(validation_summary.get("f1_std") or 0.0),
            -float(validation_summary.get("mean_f1") or 0.0),
            -float(validation_summary.get("mean_coverage") or 0.0),
            float(candidate.selected_threshold or 0.0),
        )
