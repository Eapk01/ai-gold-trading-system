"""Trading/broker workflow orchestration."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, TYPE_CHECKING

from src.broker_interface import broker_config_to_dict, create_broker_config
from src.config_utils import get_target_column
from src.research import resolve_research_defaults

if TYPE_CHECKING:
    from src.app_service import ResearchAppService


class TradingWorkflowService:
    """Trading/broker orchestration delegated from the app facade."""

    def __init__(self, service: "ResearchAppService") -> None:
        self.service = service

    def get_configuration_summary(self) -> Dict[str, Any]:
        self.service._reload_runtime_from_disk()
        research_defaults = resolve_research_defaults(self.service.config)
        summary = {
            "trading_symbol": self.service.config["trading"]["symbol"],
            "timeframe": self.service.config["trading"]["timeframe"],
            "primary_data_source": self.service.config["data_sources"]["primary"],
            "dataset_directory": self.service.config["data_sources"].get("dataset_directory", "data/imports"),
            "model_type": self.service.config["ai_model"]["type"],
            "enabled_models": list(self.service.config["ai_model"]["models"]),
            "database_path": self.service.config["database"]["path"],
            "target_column": get_target_column(self.service.config),
            "research_primary_workflow": "search",
            "research_diagnostic_workflows": [
                "single_experiment",
                "target_comparison",
                "feature_comparison",
                "candidate_training",
            ],
            "research_defaults": research_defaults.to_dict(),
            "research_stage4_default_trainer_name": research_defaults.stage4.trainer_name,
            "research_stage5_default_trainer_name": research_defaults.stage5.trainer_name,
            "research_stage5_default_target_ids": list(research_defaults.stage5.target_ids),
            "research_stage5_default_feature_sets": list(research_defaults.stage5.feature_set_names),
            "research_stage5_default_presets": list(research_defaults.stage5.preset_names),
        }
        return self.service._response(True, "Configuration loaded", data=summary)

    def get_trading_snapshot(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        bars: int = 200,
    ) -> Dict[str, Any]:
        trading_symbol = (symbol or self.service.config["trading"]["symbol"]).strip()
        trading_timeframe = (timeframe or self.service.config["trading"]["timeframe"]).strip()

        quote = self.service.broker_manager.get_market_data(trading_symbol)
        account_info = self.service.broker_manager.get_account_info()
        positions = self.service.broker_manager.get_positions()
        chart_data, chart_source, chart_reason = self.service._get_chart_data(trading_symbol, trading_timeframe, bars)

        return self.service._response(
            True,
            f"Trading snapshot loaded for {trading_symbol}",
            data={
                "symbol": trading_symbol,
                "timeframe": trading_timeframe,
                "quote": quote,
                "account": account_info,
                "positions": positions,
                "chart": chart_data,
                "chart_source": chart_source,
                "chart_reason": chart_reason,
                "broker_connected": bool(self.service.broker_manager.active_broker),
            },
        )

    def place_manual_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Dict[str, Any]:
        cleaned_symbol = symbol.strip() or self.service.config["trading"]["symbol"]
        if quantity <= 0:
            return self.service._response(False, "Quantity must be greater than zero")

        result = self.service.broker_manager.place_order(
            symbol=cleaned_symbol,
            side=side.strip().lower(),
            quantity=float(quantity),
            order_type="market",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if result.get("success"):
            return self.service._response(
                True,
                f"{side.title()} market order submitted for {cleaned_symbol}",
                data=result,
            )

        return self.service._response(
            False,
            result.get("error", f"Failed to submit {side} market order"),
            data=result,
            errors=[result.get("error")] if result.get("error") else None,
        )

    def close_manual_position(self, position_ticket: str) -> Dict[str, Any]:
        if not str(position_ticket).strip():
            return self.service._response(False, "Position ticket is required")

        result = self.service.broker_manager.close_position(str(position_ticket).strip())
        if result.get("success"):
            return self.service._response(
                True,
                f"Closed position {position_ticket}",
                data=result,
            )

        return self.service._response(
            False,
            result.get("error", f"Failed to close position {position_ticket}"),
            data=result,
            errors=[result.get("error")] if result.get("error") else None,
        )

    def save_broker_profile(
        self,
        *,
        name: str,
        login: str,
        password: str,
        server: str,
        terminal_path: str = "",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        name = name.strip()
        if not name:
            return self.service._response(False, "Profile name is required")
        if not str(password).strip():
            return self.service._response(False, "MT5 password is required")

        existing_profiles = self.service.config.get("brokers", {}).get("profiles", {})
        if name in existing_profiles and not overwrite:
            return self.service._response(False, f"A saved profile named '{name}' already exists")

        try:
            broker_config = create_broker_config(
                broker_type="exness",
                login=login.strip(),
                password="",
                server=server.strip(),
                terminal_path=terminal_path.strip(),
                sandbox=False,
            )
        except Exception as exc:
            return self.service._response(False, f"Failed to create broker profile: {exc}", errors=[str(exc)])

        success = self.service.broker_manager.add_broker(name, broker_config)
        if not success:
            return self.service._response(False, f"Failed to save broker profile: {name}")

        self.service.config.setdefault("brokers", {})
        self.service.config["brokers"].setdefault("profiles", {})
        self.service.config["brokers"]["profiles"][name] = broker_config_to_dict(broker_config)
        if not self.service.config["brokers"].get("default_profile"):
            self.service.config["brokers"]["default_profile"] = name
        self.service.secret_store.set_password(name, password)
        self.service._persist_config()

        return self.service._response(
            True,
            f"Broker profile saved securely: {name}",
            data={"name": name, "server": broker_config.server},
        )

    def connect_broker(self, name: str) -> Dict[str, Any]:
        secret = self.service.secret_store.get_password(name)
        if not secret:
            return self.service._response(False, f"Connection failed for broker: {name}. Saved secret is missing.")
        success = self.service.broker_manager.connect_broker(name, password=secret)
        if not success:
            return self.service._response(False, f"Connection failed for broker: {name}")
        self.service.config.setdefault("brokers", {})
        self.service.config["brokers"]["default_profile"] = name
        self.service._persist_config()
        return self.service._response(True, f"Connection successful for broker: {name}")

    def disconnect_all_brokers(self) -> Dict[str, Any]:
        self.service.broker_manager.disconnect_all()
        return self.service._response(True, "Disconnected all broker connections")

    def delete_broker_profile(self, name: str) -> Dict[str, Any]:
        profiles = self.service.config.setdefault("brokers", {}).setdefault("profiles", {})
        if name not in profiles:
            return self.service._response(False, f"Saved profile not found: {name}")

        del profiles[name]
        self.service.broker_manager.remove_broker(name)
        self.service.secret_store.delete_password(name)
        if self.service.config["brokers"].get("default_profile") == name:
            self.service.config["brokers"]["default_profile"] = next(iter(profiles.keys()), "")
        self.service._persist_config()
        return self.service._response(True, f"Broker profile deleted: {name}")

    def start_auto_trader(self) -> Dict[str, Any]:
        self.service._reload_runtime_from_disk()
        success, message = self.service.auto_trader.start()
        return self.service._response(success, message, data=self.service.auto_trader.get_status())

    def stop_auto_trader(self) -> Dict[str, Any]:
        success, message = self.service.auto_trader.stop()
        return self.service._response(success, message, data=self.service.auto_trader.get_status())

    def get_auto_trader_status(self) -> Dict[str, Any]:
        return self.service._response(True, "Auto trader status loaded", data=self.service.auto_trader.get_status())

    def get_auto_trader_events(self, limit: int = 20) -> Dict[str, Any]:
        return self.service._response(
            True,
            "Auto trader events loaded",
            data=self.service.auto_trader.get_recent_events(limit),
        )

    def get_auto_trader_settings_catalog(self) -> Dict[str, Any]:
        saved_values = self._extract_auto_trader_settings(self.service.config)
        session_values = self._extract_auto_trader_settings(self.service._get_effective_runtime_config())
        running = bool(self.service.auto_trader.get_status().get("running"))
        custom_presets = self._get_custom_presets()
        built_in_presets = self._get_built_in_presets(saved_values)
        selected_preset_id = self._match_preset_id(session_values, built_in_presets + custom_presets)
        return self.service._response(
            True,
            "Auto trader settings catalog loaded",
            data={
                "running": running,
                "saved_values": saved_values,
                "session_values": session_values,
                "built_in_presets": built_in_presets,
                "custom_presets": custom_presets,
                "selected_preset_id": selected_preset_id,
                "differs_from_defaults": session_values != saved_values,
                "differs_from_saved_session": bool(self.service.auto_trader_session_overrides),
            },
        )

    def apply_auto_trader_settings(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._validate_auto_trader_settings(overrides)
        if isinstance(validated, str):
            return self.service._response(False, validated)

        self.service.auto_trader_session_overrides = self._build_auto_trader_override_payload(validated)
        running = bool(self.service.auto_trader.get_status().get("running"))
        if not running:
            self.service._build_runtime_components()

        message = "Auto trader settings applied for this session"
        if running:
            message += ". Restart the auto trader for changes to take effect."
        return self.service._response(
            True,
            message,
            data={
                "session_values": self._extract_auto_trader_settings(self.service._get_effective_runtime_config()),
                "running": running,
                "restart_required": running,
            },
        )

    def save_auto_trader_settings_as_defaults(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._validate_auto_trader_settings(overrides)
        if isinstance(validated, str):
            return self.service._response(False, validated)

        self._apply_settings_to_config(self.service.config, validated)
        self.service._persist_config()
        self.service.auto_trader_session_overrides = self._build_auto_trader_override_payload(validated)

        running = bool(self.service.auto_trader.get_status().get("running"))
        if not running:
            self.service._build_runtime_components()

        message = "Auto trader defaults saved"
        if running:
            message += ". Restart the auto trader for changes to take effect."
        return self.service._response(
            True,
            message,
            data={
                "saved_values": self._extract_auto_trader_settings(self.service.config),
                "running": running,
                "restart_required": running,
            },
        )

    def save_auto_trader_preset(self, name: str, values: Dict[str, Any]) -> Dict[str, Any]:
        validated = self._validate_auto_trader_settings(values)
        if isinstance(validated, str):
            return self.service._response(False, validated)

        clean_name = self._sanitize_preset_name(name)
        if not clean_name:
            return self.service._response(False, "Preset name is required")

        built_in_ids = {preset["id"] for preset in self._get_built_in_presets(self._extract_auto_trader_settings(self.service.config))}
        if clean_name in built_in_ids:
            return self.service._response(False, f"Preset name '{clean_name}' is reserved by a built-in preset")

        presets = self.service.config.setdefault("live_trading", {}).setdefault("presets", {})
        presets[clean_name] = {
            "display_name": name.strip() or clean_name,
            "description": f"Custom preset saved on {self._current_timestamp()}",
            "values": deepcopy(validated),
        }
        self.service._persist_config()
        return self.service._response(True, f"Saved auto trader preset: {clean_name}", data={"preset_id": clean_name})

    def delete_auto_trader_preset(self, name: str) -> Dict[str, Any]:
        clean_name = self._sanitize_preset_name(name)
        presets = self.service.config.setdefault("live_trading", {}).setdefault("presets", {})
        if clean_name not in presets:
            return self.service._response(False, f"Custom preset not found: {clean_name}")
        del presets[clean_name]
        self.service._persist_config()
        return self.service._response(True, f"Deleted auto trader preset: {clean_name}")

    def apply_auto_trader_preset(self, name: str, scope: str = "session") -> Dict[str, Any]:
        preset = self._find_preset(name)
        if preset is None:
            return self.service._response(False, f"Preset not found: {name}")

        values = dict(preset.get("values") or {})
        if str(scope).strip().lower() == "session":
            return self.apply_auto_trader_settings(values)
        if str(scope).strip().lower() == "defaults":
            return self.save_auto_trader_settings_as_defaults(values)
        return self.service._response(False, f"Unsupported preset apply scope: {scope}")

    def _extract_auto_trader_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        live_trading = dict(config.get("live_trading", {}) or {})
        exit_management = dict(live_trading.get("exit_management", {}) or {})
        return {
            "stop_loss_pips": float(config.get("trading", {}).get("stop_loss_pips", 0.0)),
            "take_profit_pips": float(config.get("trading", {}).get("take_profit_pips", 0.0)),
            "signal_confidence_threshold": float(live_trading.get("signal_confidence_threshold", config.get("trading", {}).get("confidence_threshold", 0.6))),
            "exit_management_mode": str(exit_management.get("mode", "disabled")),
            "break_even_enabled": bool(exit_management.get("break_even_enabled", True)),
            "break_even_trigger_pips": float(exit_management.get("break_even_trigger_pips", 0.0)),
            "break_even_offset_pips": float(exit_management.get("break_even_offset_pips", 0.0)),
            "trailing_enabled": bool(exit_management.get("trailing_enabled", True)),
            "trailing_activation_pips": float(exit_management.get("trailing_activation_pips", 0.0)),
            "trailing_distance_pips": float(exit_management.get("trailing_distance_pips", 0.0)),
            "trailing_step_pips": float(exit_management.get("trailing_step_pips", 0.0)),
            "keep_take_profit": bool(exit_management.get("keep_take_profit", True)),
        }

    def _validate_auto_trader_settings(self, values: Dict[str, Any]) -> Dict[str, Any] | str:
        try:
            normalized = {
                "stop_loss_pips": float(values.get("stop_loss_pips", 0.0)),
                "take_profit_pips": float(values.get("take_profit_pips", 0.0)),
                "signal_confidence_threshold": float(values.get("signal_confidence_threshold", 0.0)),
                "exit_management_mode": str(values.get("exit_management_mode", "disabled")).strip().lower() or "disabled",
                "break_even_enabled": bool(values.get("break_even_enabled", True)),
                "break_even_trigger_pips": float(values.get("break_even_trigger_pips", 0.0)),
                "break_even_offset_pips": float(values.get("break_even_offset_pips", 0.0)),
                "trailing_enabled": bool(values.get("trailing_enabled", True)),
                "trailing_activation_pips": float(values.get("trailing_activation_pips", 0.0)),
                "trailing_distance_pips": float(values.get("trailing_distance_pips", 0.0)),
                "trailing_step_pips": float(values.get("trailing_step_pips", 0.0)),
                "keep_take_profit": bool(values.get("keep_take_profit", True)),
            }
        except (TypeError, ValueError):
            return "Auto trader settings contain an invalid numeric value"

        if normalized["stop_loss_pips"] <= 0:
            return "Stop loss must be greater than zero"
        if normalized["take_profit_pips"] <= 0:
            return "Take profit must be greater than zero"
        if not 0.0 <= normalized["signal_confidence_threshold"] <= 1.0:
            return "Signal confidence threshold must be between 0 and 1"
        if normalized["exit_management_mode"] not in {"disabled", "trailing_stop"}:
            return "Exit management mode must be 'disabled' or 'trailing_stop'"
        if normalized["break_even_trigger_pips"] < 0 or normalized["break_even_offset_pips"] < 0:
            return "Break-even settings must be zero or greater"
        if normalized["trailing_activation_pips"] < 0 or normalized["trailing_distance_pips"] < 0 or normalized["trailing_step_pips"] < 0:
            return "Trailing-stop settings must be zero or greater"
        return normalized

    def _build_auto_trader_override_payload(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trading": {
                "stop_loss_pips": float(values["stop_loss_pips"]),
                "take_profit_pips": float(values["take_profit_pips"]),
            },
            "live_trading": {
                "signal_confidence_threshold": float(values["signal_confidence_threshold"]),
                "exit_management": {
                    "mode": str(values["exit_management_mode"]),
                    "break_even_enabled": bool(values["break_even_enabled"]),
                    "break_even_trigger_pips": float(values["break_even_trigger_pips"]),
                    "break_even_offset_pips": float(values["break_even_offset_pips"]),
                    "trailing_enabled": bool(values["trailing_enabled"]),
                    "trailing_activation_pips": float(values["trailing_activation_pips"]),
                    "trailing_distance_pips": float(values["trailing_distance_pips"]),
                    "trailing_step_pips": float(values["trailing_step_pips"]),
                    "keep_take_profit": bool(values["keep_take_profit"]),
                },
            },
        }

    def _apply_settings_to_config(self, config: Dict[str, Any], values: Dict[str, Any]) -> None:
        config.setdefault("trading", {})
        config["trading"]["stop_loss_pips"] = float(values["stop_loss_pips"])
        config["trading"]["take_profit_pips"] = float(values["take_profit_pips"])
        config.setdefault("live_trading", {})
        config["live_trading"]["signal_confidence_threshold"] = float(values["signal_confidence_threshold"])
        config["live_trading"]["exit_management"] = {
            "mode": str(values["exit_management_mode"]),
            "break_even_enabled": bool(values["break_even_enabled"]),
            "break_even_trigger_pips": float(values["break_even_trigger_pips"]),
            "break_even_offset_pips": float(values["break_even_offset_pips"]),
            "trailing_enabled": bool(values["trailing_enabled"]),
            "trailing_activation_pips": float(values["trailing_activation_pips"]),
            "trailing_distance_pips": float(values["trailing_distance_pips"]),
            "trailing_step_pips": float(values["trailing_step_pips"]),
            "keep_take_profit": bool(values["keep_take_profit"]),
        }

    def _get_built_in_presets(self, saved_values: Dict[str, Any]) -> list[Dict[str, Any]]:
        return [
            {
                "id": "current_defaults",
                "display_name": "Current Defaults",
                "kind": "built_in",
                "description": "Matches the saved config defaults currently on disk.",
                "values": deepcopy(saved_values),
            },
            {
                "id": "loose_5m_fixed",
                "display_name": "Loose 5m Fixed",
                "kind": "built_in",
                "description": "Wider room for gold moves on 5-minute candles.",
                "values": {
                    "stop_loss_pips": 25.0,
                    "take_profit_pips": 45.0,
                    "signal_confidence_threshold": 0.60,
                    "exit_management_mode": "trailing_stop",
                    "break_even_enabled": True,
                    "break_even_trigger_pips": 8.0,
                    "break_even_offset_pips": 1.0,
                    "trailing_enabled": True,
                    "trailing_activation_pips": 12.0,
                    "trailing_distance_pips": 5.0,
                    "trailing_step_pips": 2.0,
                    "keep_take_profit": True,
                },
            },
            {
                "id": "balanced_5m_fixed",
                "display_name": "Balanced 5m Fixed",
                "kind": "built_in",
                "description": "Balanced profit protection and room for follow-through on 5-minute gold.",
                "values": {
                    "stop_loss_pips": 20.0,
                    "take_profit_pips": 35.0,
                    "signal_confidence_threshold": 0.58,
                    "exit_management_mode": "trailing_stop",
                    "break_even_enabled": True,
                    "break_even_trigger_pips": 6.0,
                    "break_even_offset_pips": 0.5,
                    "trailing_enabled": True,
                    "trailing_activation_pips": 9.0,
                    "trailing_distance_pips": 3.5,
                    "trailing_step_pips": 1.0,
                    "keep_take_profit": True,
                },
            },
            {
                "id": "tight_5m_fixed",
                "display_name": "Tight 5m Fixed",
                "kind": "built_in",
                "description": "Fast protection for short-lived 5-minute moves at the cost of more early exits.",
                "values": {
                    "stop_loss_pips": 15.0,
                    "take_profit_pips": 25.0,
                    "signal_confidence_threshold": 0.56,
                    "exit_management_mode": "trailing_stop",
                    "break_even_enabled": True,
                    "break_even_trigger_pips": 4.0,
                    "break_even_offset_pips": 0.3,
                    "trailing_enabled": True,
                    "trailing_activation_pips": 6.0,
                    "trailing_distance_pips": 2.5,
                    "trailing_step_pips": 0.8,
                    "keep_take_profit": True,
                },
            },
        ]

    def _get_custom_presets(self) -> list[Dict[str, Any]]:
        presets = self.service.config.get("live_trading", {}).get("presets", {}) or {}
        items: list[Dict[str, Any]] = []
        for preset_id, preset_payload in presets.items():
            items.append(
                {
                    "id": str(preset_id),
                    "display_name": str((preset_payload or {}).get("display_name") or preset_id),
                    "kind": "custom",
                    "description": str((preset_payload or {}).get("description") or "Custom preset"),
                    "values": self._extract_auto_trader_settings(self.service._deep_merge_dicts(self.service.config, self._build_auto_trader_override_payload((preset_payload or {}).get("values") or {}))),
                }
            )
        return sorted(items, key=lambda item: item["display_name"].lower())

    def _find_preset(self, preset_id: str) -> Dict[str, Any] | None:
        target = str(preset_id).strip()
        for preset in self._get_built_in_presets(self._extract_auto_trader_settings(self.service.config)) + self._get_custom_presets():
            if preset["id"] == target:
                return preset
        return None

    def _match_preset_id(self, values: Dict[str, Any], presets: list[Dict[str, Any]]) -> str | None:
        for preset in presets:
            if dict(preset.get("values") or {}) == values:
                return str(preset["id"])
        return None

    def _sanitize_preset_name(self, name: str) -> str:
        clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
        return clean_name.strip("_")

    def _current_timestamp(self) -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
