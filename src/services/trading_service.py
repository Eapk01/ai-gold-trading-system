"""Trading/broker workflow orchestration."""

from __future__ import annotations

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
