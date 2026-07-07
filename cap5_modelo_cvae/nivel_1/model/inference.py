"""Pump failure prediction - cond_reg_v2 production inference.

Self-contained module for running predictions with the trained Temporal
Conditional VAE regressor. No training dependencies are required
(no Lightning, Optuna, or scikit-learn).

Usage
-----
::

    from ensemble.cond_reg_v2.model.inference import PumpPredictor

    predictor = PumpPredictor()
    predictions = predictor.predict(df)

Input DataFrame columns
    - "Ambient temperature"
    - "Main HTF Pump Speed"
    - "Main HTF Pump Inlet Temperature"
    - "pump_id" (int 1-4) OR one-hot "pump_id_1" ... "pump_id_4"

Output DataFrame
    - Predicted output variables defined by the training checkpoint
      (typically 13 sensor columns), denormalized to physical units.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = ["PumpPredictor"]

WEIGHTS_DIR = Path(__file__).parent / "weights"


class _Layer(nn.Module):
    """FC -> optional BN -> LeakyReLU(0.1) -> optional Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _TemporalEmbedding(nn.Module):
    """Shared MLP per timestep plus learned positional encoding."""

    def __init__(
        self,
        n_input: int,
        embed_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            _Layer(n_input, 128, bn=True, dropout=dropout),
            _Layer(128, embed_dim, bn=True, dropout=dropout),
        )
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_input]
        batch_size, seq_len, n_input = x.shape
        x = x.reshape(batch_size * seq_len, n_input)
        x = self.embed(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = x + self.positional_encoding[:, :seq_len, :]
        return x


class _TemporalAttention(nn.Module):
    """Self-attention over timesteps with residual normalization and pooling."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.mha(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.norm(x + attn_out)

        # Use last query attention over full history, averaged across heads.
        pooled_weights = attn_weights.mean(dim=1)[:, -1, :]
        pooled = (pooled_weights.unsqueeze(-1) * attended).sum(dim=1)
        return pooled, attn_weights


class _Encoder(nn.Module):
    """Temporal pooled embedding to latent distribution parameters."""

    def __init__(
        self,
        embed_dim: int,
        layer_sizes: list[int],
        latent_dim: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()
        dims = [embed_dim] + layer_sizes
        self.layers = nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], batch_norm, dropout) for i in range(len(dims) - 1)],
        )
        self.mu = nn.Linear(dims[-1], latent_dim)
        self.logvar = nn.Linear(dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.layers(x)
        return self.mu(h), self.logvar(h), h


class _Decoder(nn.Module):
    """Latent mean to current-timestep output predictions."""

    def __init__(
        self,
        latent_dim: int,
        layer_sizes: list[int],
        output_dim: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()
        dims = [latent_dim] + list(reversed(layer_sizes))
        self.layers = nn.Sequential(
            *[_Layer(dims[i], dims[i + 1], batch_norm, dropout) for i in range(len(dims) - 1)],
        )
        self.reconstructed = nn.Linear(dims[-1], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.reconstructed(self.layers(z))


class PumpPredictor:
    """
    Predict pump sensor outputs from operating conditions.

    Loads the trained Temporal Conditional VAE weights and normalisation
    parameters. All preprocessing and post-processing handled internally.

    Parameters
    ----------
    weights_dir : str or Path, optional
        Directory containing best_weights.pt and norm_params.json.
        Defaults to the weights/ folder shipped with this package.
    device : str, optional
        "cpu", "cuda", "mps", or "auto" (default).
    """

    def __init__(self, weights_dir: str | Path | None = None, device: str = "auto"):
        self._weights_dir = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
        ckpt_path = self._validate_weights_dir(self._weights_dir)

        checkpoint = self._load_checkpoint_payload(ckpt_path)
        with open(self._weights_dir / "norm_params.json", "r", encoding="utf-8") as f:
            self._norm_params: dict = json.load(f)

        self._params: dict = checkpoint["params"]
        self._input_vars: list[str] = list(self._params["input_variables"])
        self._output_vars: list[str] = list(self._params["output_variables"])
        self._past_history = int(self._params["past_history"])
        self._n_input = len(self._input_vars)
        self._output_dim = len(self._output_vars)

        layer_sizes = [int(size) for size in str(self._params["layer_sizes"]).split(",")]
        latent_dim = int(self._params["latent_dim"])
        embed_dim = int(self._params["embed_dim"])
        n_attention_heads = int(self._params["n_attention_heads"])
        batch_norm = bool(self._params.get("batch_norm", True))
        dropout = float(self._params.get("dropout", 0.0))

        self._temporal_embedding = _TemporalEmbedding(
            n_input=self._n_input,
            embed_dim=embed_dim,
            dropout=dropout,
            max_seq_len=max(self._past_history, 10),
        )
        self._temporal_attention = _TemporalAttention(
            embed_dim=embed_dim,
            n_heads=n_attention_heads,
            dropout=dropout,
        )
        self._encoder = _Encoder(
            embed_dim=embed_dim,
            layer_sizes=layer_sizes,
            latent_dim=latent_dim,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self._decoder = _Decoder(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            output_dim=self._output_dim,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        self._temporal_embedding.load_state_dict(checkpoint["temporal_embedding_state_dict"])
        self._temporal_attention.load_state_dict(checkpoint["temporal_attention_state_dict"])
        self._encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self._decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self._device = self._resolve_device(device)
        self._temporal_embedding.to(self._device).eval()
        self._temporal_attention.to(self._device).eval()
        self._encoder.to(self._device).eval()
        self._decoder.to(self._device).eval()

        for module in (
            self._temporal_embedding,
            self._temporal_attention,
            self._encoder,
            self._decoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def input_columns(self) -> list[str]:
        """Expected input feature names (after one-hot encoding)."""
        return list(self._input_vars)

    @property
    def output_columns(self) -> list[str]:
        """Predicted output feature names."""
        return list(self._output_vars)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict pump output variables from operating conditions.

        Input DataFrame must contain:
            - "Ambient temperature"
            - "Main HTF Pump Speed"
            - "Main HTF Pump Inlet Temperature"
            - "pump_id" (int 1-4) OR one-hot "pump_id_1"..."pump_id_4"

        Returns DataFrame with predicted sensor columns,
        denormalized to physical units, same index as input.
        """
        preprocessed = self._preprocess(df)
        normalized_inputs = self._normalize(preprocessed[self._input_vars])
        windows_flat = self._build_windows(normalized_inputs)

        with torch.no_grad():
            pred_norm, _ = self._forward_from_windows(windows_flat)

        pred_norm_df = pd.DataFrame(pred_norm, columns=self._output_vars, index=df.index)
        return self._denormalize(pred_norm_df)

    def predict_with_interpretability(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
        """
        Enhanced prediction with attention weights and feature importance.

        Returns:
            predictions_df: DataFrame with predicted output sensor columns
            attention_weights: numpy array [N, past_history] - temporal attention
            feature_contributions: dict mapping feature name -> importance score
        """
        preprocessed = self._preprocess(df)
        normalized_inputs = self._normalize(preprocessed[self._input_vars])
        windows_flat = self._build_windows(normalized_inputs)

        with torch.no_grad():
            pred_norm, pooled_attention = self._forward_from_windows(windows_flat)

        pred_norm_df = pd.DataFrame(pred_norm, columns=self._output_vars, index=df.index)
        predictions_df = self._denormalize(pred_norm_df)

        feature_contributions = self._feature_contributions_from_windows(
            windows_flat=windows_flat,
            pooled_attention=pooled_attention,
        )
        return predictions_df, pooled_attention, feature_contributions

    @staticmethod
    def _validate_weights_dir(weights_dir: Path) -> Path:
        if not weights_dir.exists() or not weights_dir.is_dir():
            raise FileNotFoundError(f"weights_dir not found: {weights_dir}")

        if not (weights_dir / "norm_params.json").exists():
            raise FileNotFoundError(
                f"weights_dir missing required files: norm_params.json (weights_dir={weights_dir})",
            )

        best_weights = weights_dir / "best_weights.pt"
        lightning_ckpt = weights_dir / "model_weights.ckpt"

        if best_weights.exists():
            return best_weights
        if lightning_ckpt.exists():
            return lightning_ckpt

        raise FileNotFoundError(
            "weights_dir must contain either best_weights.pt or model_weights.ckpt "
            f"(weights_dir={weights_dir})",
        )

    @staticmethod
    def _extract_prefixed_state_dict(
        full_state_dict: Mapping[str, torch.Tensor],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        prefix_dot = f"{prefix}."
        out = {
            key[len(prefix_dot) :]: value
            for key, value in full_state_dict.items()
            if key.startswith(prefix_dot)
        }
        if not out:
            raise KeyError(f"No keys found for prefix '{prefix_dot}' in checkpoint state_dict")
        return out

    def _load_checkpoint_payload(self, checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if checkpoint_path.name == "best_weights.pt":
            required = {
                "params",
                "temporal_embedding_state_dict",
                "temporal_attention_state_dict",
                "encoder_state_dict",
                "decoder_state_dict",
            }
            missing = [key for key in sorted(required) if key not in checkpoint]
            if missing:
                raise KeyError(
                    f"best_weights.pt missing required keys: {missing} (path={checkpoint_path})"
                )
            return checkpoint

        if checkpoint_path.suffix == ".ckpt":
            if "state_dict" not in checkpoint:
                raise KeyError(f"Lightning checkpoint missing state_dict (path={checkpoint_path})")

            hparams = checkpoint.get("hyper_parameters")
            if not isinstance(hparams, dict):
                raise KeyError(
                    f"Lightning checkpoint missing hyper_parameters dict (path={checkpoint_path})"
                )

            state_dict = checkpoint["state_dict"]
            return {
                "params": hparams,
                "temporal_embedding_state_dict": self._extract_prefixed_state_dict(
                    state_dict, "temporal_embedding"
                ),
                "temporal_attention_state_dict": self._extract_prefixed_state_dict(
                    state_dict, "temporal_attention"
                ),
                "encoder_state_dict": self._extract_prefixed_state_dict(state_dict, "encoder"),
                "decoder_state_dict": self._extract_prefixed_state_dict(state_dict, "decoder"),
            }

        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return resolved

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode pump_id and select expected input columns."""
        out = df.copy()
        if "pump_id" in out.columns:
            for pid in (1, 2, 3, 4):
                out[f"pump_id_{pid}"] = (out["pump_id"] == pid).astype(int)
            out = out.drop(columns=["pump_id"])

        missing = [column for column in self._input_vars if column not in out.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")

        return out

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize using stored training parameters."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            stats = self._norm_params[col]
            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val
            if abs(denom) < 1e-12:
                result[col] = 0.0
            else:
                result[col] = (df[col] - min_val) / denom
        return result

    def _denormalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse min-max normalization for output columns."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            stats = self._norm_params[col]
            min_val, max_val = stats["min"], stats["max"]
            result[col] = df[col] * (max_val - min_val) + min_val
        return result

    def _build_windows(self, normalized_df: pd.DataFrame) -> torch.Tensor:
        """Build [N, past_history * n_input] windows with repeat-padding."""
        values = normalized_df[self._input_vars].to_numpy(dtype=np.float32)
        if values.shape[0] == 0:
            raise ValueError("Input DataFrame is empty")

        windows = []
        for i in range(len(values)):
            if i < self._past_history - 1:
                n_pad = self._past_history - 1 - i
                pad = np.tile(values[0], (n_pad, 1))
                window = np.vstack([pad, values[: i + 1]])
            else:
                window = values[i - self._past_history + 1 : i + 1]
            windows.append(window.reshape(-1))

        return torch.tensor(np.asarray(windows), dtype=torch.float32, device=self._device)

    def _forward_from_windows(self, windows_flat: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run full temporal inference and return normalized predictions and pooled attention."""
        x_reshaped = windows_flat.reshape(windows_flat.shape[0], self._past_history, self._n_input)
        embedded = self._temporal_embedding(x_reshaped)
        pooled, attn_weights = self._temporal_attention(embedded)
        mu, _, _ = self._encoder(pooled)
        predictions = self._decoder(mu)

        pooled_attention = attn_weights.mean(dim=1)[:, -1, :]
        return predictions.cpu().numpy(), pooled_attention.cpu().numpy()

    def _feature_contributions_from_windows(
        self,
        windows_flat: torch.Tensor,
        pooled_attention: np.ndarray,
    ) -> dict[str, float]:
        """Estimate feature importance from attention-weighted temporal variation."""
        windows_np = windows_flat.detach().cpu().numpy().reshape(-1, self._past_history, self._n_input)

        # Magnitude relative to current timestep captures temporal contribution.
        reference = windows_np[:, -1:, :]
        temporal_residual = np.abs(windows_np - reference)
        weighted = temporal_residual * pooled_attention[:, :, None]
        importance = weighted.mean(axis=(0, 1))

        total = float(np.sum(importance))
        if total > 1e-12:
            importance = importance / total

        return {
            feature: float(score)
            for feature, score in zip(self._input_vars, importance)
        }
