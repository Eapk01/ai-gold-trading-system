from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticMarketConfig:
    n_rows: int = 50000
    start_price: float = 2000.0
    freq_minutes: int = 30
    seed: int = 42

    # Controls how detectable the planted edge is.
    # 0.0 = no edge, 1.0 = strong but still noisy.
    edge_strength: float = 0.8

    # Noise level in returns.
    noise_std: float = 0.0035

    # Regime persistence: higher = longer trending/reverting patches.
    regime_flip_prob: float = 0.03

    # Output path
    output_path: str = "synthetic_xauusd.tsv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_synthetic_market(cfg: SyntheticMarketConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    n = cfg.n_rows
    dt = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n,
        freq=f"{cfg.freq_minutes}min",
    )

    # Hidden regime:
    # +1 trending-up regime
    # -1 mean-reverting/down-pressure regime
    regime = np.empty(n, dtype=np.int8)
    regime[0] = 1 if rng.random() < 0.5 else -1
    for i in range(1, n):
        if rng.random() < cfg.regime_flip_prob:
            regime[i] = -regime[i - 1]
        else:
            regime[i] = regime[i - 1]

    close = np.empty(n, dtype=np.float64)
    open_ = np.empty(n, dtype=np.float64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    tickvol = np.empty(n, dtype=np.int64)
    vol = np.zeros(n, dtype=np.int64)
    spread = np.empty(n, dtype=np.int64)

    close[0] = cfg.start_price
    open_[0] = cfg.start_price

    # Stateful features that resemble market structure a bit
    momentum_1 = np.zeros(n)
    momentum_3 = np.zeros(n)
    zscore_8 = np.zeros(n)
    volume_shock = np.zeros(n)
    spread_state = np.zeros(n)

    raw_returns = np.zeros(n)

    # First pass: generate candles with plausible-ish noise
    for i in range(1, n):
        open_[i] = close[i - 1]

        recent_ret = 0.0 if i < 2 else (close[i - 1] - close[i - 2]) / max(close[i - 2], 1e-9)
        recent_ret3 = 0.0
        if i >= 4:
            recent_ret3 = (close[i - 1] - close[i - 4]) / max(close[i - 4], 1e-9)

        recent_window = close[max(0, i - 8):i]
        ma8 = recent_window.mean() if len(recent_window) > 0 else close[i - 1]
        std8 = recent_window.std() if len(recent_window) > 1 else 1e-6
        z8 = (close[i - 1] - ma8) / max(std8, 1e-6)

        vol_shock = rng.normal(0.0, 1.0)
        spread_state[i] = 0.85 * spread_state[i - 1] + 0.15 * rng.normal()

        momentum_1[i] = recent_ret
        momentum_3[i] = recent_ret3
        zscore_8[i] = z8
        volume_shock[i] = vol_shock

        # Base current-candle movement: mostly noise + weak regime effect
        drift = 0.00015 * regime[i]
        microstructure = (
            0.00045 * regime[i] * np.tanh(20 * recent_ret)
            - 0.00025 * (1 - (regime[i] == 1)) * np.tanh(z8)
            + 0.00012 * np.tanh(vol_shock)
        )
        eps = rng.normal(0.0, cfg.noise_std)

        r_t = drift + microstructure + eps
        raw_returns[i] = r_t

        close[i] = max(1.0, open_[i] * (1.0 + r_t))

        # Construct candle range around open/close
        body_hi = max(open_[i], close[i])
        body_lo = min(open_[i], close[i])

        wick_up = abs(rng.normal(0.0, 0.0018)) * open_[i]
        wick_dn = abs(rng.normal(0.0, 0.0018)) * open_[i]

        high[i] = body_hi + wick_up
        low[i] = max(0.01, body_lo - wick_dn)

        # Tick volume correlated with volatility and shock
        abs_move = abs(r_t)
        tickvol[i] = int(
            max(
                20,
                120
                + 50000 * abs_move
                + 22 * abs(vol_shock)
                + 10 * abs(spread_state[i])
                + rng.normal(0, 8),
            )
        )

        # Integer "spread" like MT5 export
        spread[i] = int(
            max(
                1,
                15
                + 12 * abs(spread_state[i])
                + 3 * abs(z8)
                + 2 * abs(vol_shock)
                + rng.normal(0, 1.5),
            )
        )

    # Second pass:
    # Plant a NEXT-CANDLE edge using only information available at time t.
    # This affects whether the next candle closes above/below its open.
    #
    # We do this by nudging close[t+1] after building features at time t.
    for t in range(10, n - 1):
        # Features known at time t
        m1 = momentum_1[t]
        m3 = momentum_3[t]
        z8 = zscore_8[t]
        vs = volume_shock[t]
        sp = spread[t]
        reg = regime[t]

        # Score intentionally built from CURRENT and PAST data only
        # Trending regime prefers continuation, reverting regime prefers snapback.
        score = (
            2.2 * reg * np.tanh(35 * m1)
            + 1.5 * reg * np.tanh(18 * m3)
            - 1.4 * (reg == -1) * np.tanh(0.9 * z8)
            + 0.45 * np.tanh(0.25 * vs)
            - 0.02 * (sp - 15)
        )

        # Edge enters probabilistically. Still noisy.
        p_up = sigmoid(cfg.edge_strength * score)

        # Clip away from certainty
        p_up = float(np.clip(p_up, 0.12, 0.88))
        next_up = rng.random() < p_up

        # Rebuild next candle close with planted direction bias
        nxt = t + 1
        o = open_[nxt]

        # Preserve roughly similar magnitude but control sign probabilistically
        base_mag = max(abs((close[nxt] - open_[nxt]) / max(open_[nxt], 1e-9)), 0.0004)
        noisy_mag = base_mag + abs(rng.normal(0.0, 0.0012))

        signed_ret = noisy_mag if next_up else -noisy_mag
        close[nxt] = max(1.0, o * (1.0 + signed_ret))

        body_hi = max(o, close[nxt])
        body_lo = min(o, close[nxt])
        wick_up = abs(rng.normal(0.0, 0.0016)) * o
        wick_dn = abs(rng.normal(0.0, 0.0016)) * o
        high[nxt] = body_hi + wick_up
        low[nxt] = max(0.01, body_lo - wick_dn)

    df = pd.DataFrame(
        {
            "<DATE>": dt.strftime("%Y.%m.%d"),
            "<TIME>": dt.strftime("%H:%M:%S"),
            "<OPEN>": np.round(open_, 5),
            "<HIGH>": np.round(high, 5),
            "<LOW>": np.round(low, 5),
            "<CLOSE>": np.round(close, 5),
            "<TICKVOL>": tickvol,
            "<VOL>": vol,
            "<SPREAD>": spread,
        }
    )

    return df


def main() -> None:
    # Positive control: planted edge
    edge_cfg = SyntheticMarketConfig(
        n_rows=50000,
        seed=42,
        edge_strength=0.8,
        output_path="synthetic_edge.csv",
    )
    edge_df = build_synthetic_market(edge_cfg)
    edge_df.to_csv(edge_cfg.output_path, sep="\t", index=False)

    # Null control: no planted edge
    null_cfg = SyntheticMarketConfig(
        n_rows=50000,
        seed=43,
        edge_strength=0.0,
        output_path="synthetic_null.csv",
    )
    null_df = build_synthetic_market(null_cfg)
    null_df.to_csv(null_cfg.output_path, sep="\t", index=False)

    print(f"Wrote {edge_cfg.output_path} with planted edge.")
    print(f"Wrote {null_cfg.output_path} with no edge.")


if __name__ == "__main__":
    main()