"""Research/model workflow orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.app_service import ResearchAppService


class ResearchWorkflowService:
    """Research/model orchestration delegated from the app facade."""

    def __init__(self, service: "ResearchAppService") -> None:
        self.service = service

    def import_and_prepare_data(self) -> Dict[str, Any]:
        logger.info("Starting local dataset import and preparation...")
        try:
            csv_path, historical_data = self.service.data_collector.import_default_dataset()
            is_valid, issues = self.service.data_collector.validate_data_quality(historical_data)

            if historical_data.empty:
                logger.error("Imported dataset is empty after normalization")
                return self.service._response(False, "Imported dataset is empty after normalization")

            self.service.data_collector.save_data_to_db(historical_data, "raw_data")
            self.service.feature_data = self.service.feature_engineer.create_feature_matrix(
                historical_data,
                include_targets=True,
            )

            if self.service.feature_data.empty:
                logger.error("Failed to create features")
                return self.service._response(False, "Failed to create features")

            target_column = self.service.get_target_column()
            self.service.selected_features = self.service.feature_engineer.select_features(
                self.service.feature_data,
                target_column=target_column,
                method="correlation",
                max_features=30,
            )

            if not self.service.selected_features:
                logger.error("Feature selection failed")
                return self.service._response(False, "Feature selection failed")

            self.service.last_import_summary = {
                "path": str(csv_path),
                "rows": len(historical_data),
                "selected_features": len(self.service.selected_features),
                "data_valid": is_valid,
                "issues": issues,
                "feature_rows": len(self.service.feature_data),
                "target_column": target_column,
            }
            preview_frame = historical_data.reset_index().head(20)
            self.service.latest_data_preview = preview_frame.to_dict(orient="records")

            logger.info(
                f"Local dataset preparation complete - {len(historical_data)} raw rows, "
                f"{len(self.service.selected_features)} selected features"
            )
            return self.service._response(
                True,
                f"Imported {len(historical_data)} rows and selected {len(self.service.selected_features)} features",
                data={
                    "summary": self.service.last_import_summary,
                    "selected_features": list(self.service.selected_features),
                    "preview": self.service.latest_data_preview,
                },
                errors=list(issues),
            )
        except Exception as exc:
            logger.error(f"Failed to import local dataset: {exc}")
            return self.service._response(False, f"Local dataset import failed: {exc}", errors=[str(exc)])

    def train_models(self, model_name: str = "default") -> Dict[str, Any]:
        if self.service.feature_data is None or not self.service.selected_features:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        logger.info("Starting AI model training...")
        training_results = self.service.ai_model_manager.train_ensemble_models(
            self.service.feature_data,
            self.service.selected_features,
            target_column=self.service.get_target_column(),
        )

        if not training_results:
            logger.error("Model training failed")
            return self.service._response(False, "Model training failed")

        safe_name = self.service._sanitize_model_name(model_name or "default")
        model_path = Path("models") / f"{safe_name}.joblib"
        self.service.ai_model_manager.save_models(model_path)
        self.service.loaded_model_path = str(model_path)
        self.service.selected_features = list(self.service.ai_model_manager.feature_columns)
        self.service.latest_training_results = training_results

        return self.service._response(
            True,
            f"Training complete. Saved model file: {model_path.name}",
            data={
                "training_results": training_results,
                "saved_model_name": model_path.stem,
                "saved_model_path": str(model_path),
                "selected_features": list(self.service.selected_features),
            },
            artifacts={"model_path": str(model_path)},
        )

    def run_backtest(self) -> Dict[str, Any]:
        logger.info("Starting professional backtest...")
        self.service._reload_runtime_from_disk()

        if not self.service.ai_model_manager.models:
            logger.error("Please train or load the models first")
            return self.service._response(False, "Please train or load the models first")
        if self.service.feature_data is None or not self.service.selected_features:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        try:
            result = self.service.backtester.run_backtest(
                self.service.feature_data,
                self.service.ai_model_manager,
                self.service.selected_features,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/backtest_result_{timestamp}.json"
            chart_file = f"reports/backtest_chart_{timestamp}.png"
            summary_file = ""

            self.service.backtester.save_results(result, result_file)
            self.service.backtester.plot_results(result, chart_file)

            trade_summary = self.service.backtester.get_trade_summary()
            if not trade_summary.empty:
                summary_file = f"reports/trade_summary_{timestamp}.csv"
                trade_summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
                logger.info(f"Trade summary saved: {summary_file}")

            result_summary = self.service._serialize_backtest_result(result)
            artifacts = {
                "report_file": result_file,
                "chart_file": chart_file,
            }
            if summary_file:
                artifacts["trade_summary_file"] = summary_file

            self.service.latest_backtest_summary = result_summary
            self.service.latest_backtest_artifacts = artifacts

            return self.service._response(
                True,
                "Backtest completed successfully",
                data={
                    "summary": result_summary,
                    "trade_summary_rows": len(trade_summary),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Professional backtest failed: {exc}")
            return self.service._response(False, f"Professional backtest failed: {exc}", errors=[str(exc)])

    def run_model_test(self) -> Dict[str, Any]:
        logger.info("Starting model test...")
        self.service._reload_runtime_from_disk()

        if not self.service.ai_model_manager.models:
            logger.error("Please train or load the models first")
            return self.service._response(False, "Please train or load the models first")
        if self.service.feature_data is None or not self.service.selected_features:
            logger.error("Please prepare the data first")
            return self.service._response(False, "Please prepare the data first")

        target_column = self.service.ai_model_manager.target_column or self.service.get_target_column()
        if target_column not in self.service.feature_data.columns:
            message = f"Prepared feature data is missing target column: {target_column}"
            logger.error(message)
            return self.service._response(False, message)

        try:
            prediction_frame = self.service.ai_model_manager.predict_ensemble_batch(
                self.service.feature_data,
                feature_columns=self.service.selected_features,
                method="voting",
            )
            result = self.service.model_tester.evaluate(
                self.service.feature_data,
                prediction_frame,
                self.service.feature_data[target_column],
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"reports/model_test_result_{timestamp}.json"
            rows_file = f"reports/model_test_rows_{timestamp}.csv"

            threshold_rows = result.threshold_performance.to_dict(orient="records")
            confidence_bucket_rows = result.confidence_buckets.to_dict(orient="records")
            result.row_evaluations.to_csv(rows_file, index=False, encoding="utf-8-sig")

            report_payload = {
                "model_test_summary": result.summary,
                "threshold_performance": threshold_rows,
                "confidence_buckets": confidence_bucket_rows,
                "target_column": target_column,
                "selected_features": list(self.service.selected_features),
                "loaded_model_path": self.service.loaded_model_path,
                "row_evaluations_file": rows_file,
            }
            with open(result_file, "w", encoding="utf-8") as report_file:
                json.dump(report_payload, report_file, ensure_ascii=False, indent=2)

            result_summary = self.service._serialize_model_test_result(result)
            artifacts = {
                "report_file": result_file,
                "evaluation_rows_file": rows_file,
            }
            self.service.latest_model_test_summary = result_summary
            self.service.latest_model_test_artifacts = artifacts

            return self.service._response(
                True,
                "Model test completed successfully",
                data={
                    "summary": result_summary,
                    "threshold_performance": threshold_rows,
                    "confidence_buckets": confidence_bucket_rows,
                    "evaluation_preview": result.row_evaluations.head(200).to_dict(orient="records"),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            logger.error(f"Model test failed: {exc}")
            return self.service._response(False, f"Model test failed: {exc}", errors=[str(exc)])

    def get_model_analysis(self) -> Dict[str, Any]:
        if not self.service.ai_model_manager.models:
            return self.service._response(False, "Please train or load the models first")

        summary = self.service.ai_model_manager.get_models_summary()
        feature_importance = {}
        for model_name in self.service.ai_model_manager.models:
            importance = self.service.ai_model_manager.get_feature_importance(model_name)
            if importance:
                feature_importance[model_name] = list(importance.items())[:10]

        self.service.latest_model_analysis = {
            "summary": summary.to_dict(orient="records") if not summary.empty else [],
            "feature_importance": feature_importance,
        }
        return self.service._response(True, "Model analysis loaded", data=self.service.latest_model_analysis)
