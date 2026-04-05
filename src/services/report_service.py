"""Report listing/loading workflow helpers."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from loguru import logger

from src.report_store import ReportDefinition, ReportStore

if TYPE_CHECKING:
    from src.app_service import ResearchAppService


BACKTEST_REPORT = ReportDefinition("backtest", "backtest_result", "backtest_summary")
MODEL_TEST_REPORT = ReportDefinition("model_test", "model_test_result", "model_test_summary")


class ReportWorkflowService:
    """Shared report listing/loading helpers."""

    def __init__(self, service: "ResearchAppService", store: ReportStore) -> None:
        self.service = service
        self.store = store

    def list_backtest_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = self.store.list_reports(BACKTEST_REPORT, limit=limit)
        message = "Backtest reports loaded" if reports else "No backtest report files found"
        return self.service._response(True, message, data=reports)

    def get_backtest_report(self, report_path: str) -> Dict[str, Any]:
        try:
            report_data = self.store.load_report(BACKTEST_REPORT, report_path)
            payload = report_data["report_payload"]
            return self.service._response(
                True,
                f"Loaded backtest report: {report_data['name']}",
                data={
                    "report_type": report_data["report_type"],
                    "path": report_path,
                    "summary": report_data["summary"],
                    "trade_count": len(payload.get("trades", [])),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load backtest report: {exc}")
            return self.service._response(False, f"Failed to load backtest report: {exc}", errors=[str(exc)])

    def list_model_test_reports(self, limit: int = 10) -> Dict[str, Any]:
        reports = self.store.list_reports(MODEL_TEST_REPORT, limit=limit)
        message = "Model test reports loaded" if reports else "No model test report files found"
        return self.service._response(True, message, data=reports)

    def get_model_test_report(self, report_path: str) -> Dict[str, Any]:
        try:
            report_data = self.store.load_report(MODEL_TEST_REPORT, report_path)
            payload = report_data["report_payload"]
            return self.service._response(
                True,
                f"Loaded model test report: {report_data['name']}",
                data={
                    "report_type": report_data["report_type"],
                    "path": report_path,
                    "summary": report_data["summary"],
                    "threshold_performance": payload.get("threshold_performance", []),
                    "confidence_buckets": payload.get("confidence_buckets", []),
                    "row_evaluations_file": payload.get("row_evaluations_file"),
                    "target_column": payload.get("target_column"),
                },
            )
        except Exception as exc:
            logger.error(f"Failed to load model test report: {exc}")
            return self.service._response(False, f"Failed to load model test report: {exc}", errors=[str(exc)])
