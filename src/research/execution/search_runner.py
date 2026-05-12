"""Bounded research search orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from ..catalog.search_presets import expand_lstm_search_preset_variants
from ..schemas import SearchCandidateSummary, SearchRequest
from ..search_truth_gate import apply_truth_gate, build_truth_gate_summary


@dataclass(frozen=True)
class SearchCandidateConfig:
    """One bounded research search candidate configuration."""

    candidate_id: str
    target_spec: Dict[str, Any]
    target_spec_id: str
    target_display_name: str
    feature_set_name: str
    preset_name: str
    preset_variant_name: str
    preset_variant_summary: str
    trainer_name: str
    trainer_params: Dict[str, Any]


@dataclass
class SearchRunner:
    """Build and rank bounded research search candidates."""

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
                    preset_payload = dict(preset_definitions[preset_name])
                    variant_payloads = (
                        expand_lstm_search_preset_variants(preset_name, preset_payload)
                        if str(request.trainer_name or "").strip().lower() == "lstm"
                        else [
                            {
                                **preset_payload,
                                "preset_variant_name": preset_name,
                                "preset_variant_summary": preset_name,
                            }
                        ]
                    )
                    for variant_payload in variant_payloads:
                        is_lstm = str(request.trainer_name or "").strip().lower() == "lstm"
                        preset_variant_name = str(variant_payload.get("preset_variant_name") or preset_name)
                        trainer_params = {
                            key: value
                            for key, value in dict(variant_payload).items()
                            if key not in {"preset_name", "preset_variant_name", "preset_variant_summary"}
                        }
                        preset_variant_summary = str(
                            variant_payload.get("preset_variant_summary")
                            or self._build_variant_summary(trainer_params)
                        )
                        candidates.append(
                            SearchCandidateConfig(
                                candidate_id=(
                                    f"{request.search_id}_{target_spec_id}_{feature_set_name}_{preset_name}_{preset_variant_name}"
                                    if is_lstm
                                    else f"{request.search_id}_{target_spec_id}_{feature_set_name}_{preset_name}"
                                ),
                                target_spec=dict(target_spec),
                                target_spec_id=target_spec_id,
                                target_display_name=target_display_name,
                                feature_set_name=feature_set_name,
                                preset_name=preset_name,
                                preset_variant_name=preset_variant_name,
                                preset_variant_summary=preset_variant_summary,
                                trainer_name=request.trainer_name,
                                trainer_params=trainer_params,
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
                str(candidate.preset_variant_name or ""),
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
            "reason": "No candidate passed the full research truth gate.",
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
                    "preset_variant_name": candidate.preset_variant_name,
                    "preset_variant_summary": candidate.preset_variant_summary,
                    "selected_threshold": candidate.selected_threshold,
                    "threshold_source": candidate.threshold_source,
                    "architecture_name": candidate.architecture_name,
                    "feature_mode": candidate.feature_mode,
                    "sequence_feature_count": candidate.sequence_feature_count,
                    "dense_head_summary": candidate.dense_head_summary,
                    "bidirectional": candidate.bidirectional,
                    "training_device": candidate.training_device,
                    "cuda_available": candidate.cuda_available,
                    "cuda_device_name": candidate.cuda_device_name,
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
                "preset_variant_name": recommended_candidate.preset_variant_name,
                "preset_variant_summary": recommended_candidate.preset_variant_summary,
                "selected_threshold": recommended_candidate.selected_threshold,
                "threshold_source": recommended_candidate.threshold_source,
                "architecture_name": recommended_candidate.architecture_name,
                "feature_mode": recommended_candidate.feature_mode,
                "sequence_feature_count": recommended_candidate.sequence_feature_count,
                "dense_head_summary": recommended_candidate.dense_head_summary,
                "bidirectional": recommended_candidate.bidirectional,
                "training_device": recommended_candidate.training_device,
                "cuda_available": recommended_candidate.cuda_available,
                "cuda_device_name": recommended_candidate.cuda_device_name,
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

    @staticmethod
    def _build_variant_summary(trainer_params: Dict[str, Any]) -> str:
        if "feature_mode" not in trainer_params:
            return str(trainer_params.get("preset_variant_name") or "")
        parts = [
            str(trainer_params.get("feature_mode") or "engineered"),
            f"lookback {trainer_params.get('lookback_window')}",
            f"hidden {trainer_params.get('hidden_size')}",
            f"dense {trainer_params.get('dense_hidden_size')}",
            f"lr {trainer_params.get('learning_rate')}",
        ]
        if bool(trainer_params.get("bidirectional")):
            parts.append("bidirectional")
        return " | ".join(parts)
