"""Temporal Conditional VAE model for cond_reg_v2.

This module predicts only the current timestep outputs from a short window of
past condition inputs. A temporal attention block keeps interpretability by
exposing attention maps over the history window.
"""

from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .device import should_pin_memory


class PumpDataset(Dataset):
    """Simple tensor dataset for (x, y) pairs."""

    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.x_data.shape[0]


class Layer(nn.Module):
    """FC -> optional BN -> LeakyReLU -> optional Dropout block."""

    def __init__(self, in_dim, out_dim, bn=True, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)
        init.xavier_uniform_(self.block[0].weight)

    def forward(self, x):
        return self.block(x)


class TemporalEmbedding(nn.Module):
    """Shared MLP per timestep + learned positional encoding."""

    def __init__(self, n_input, embed_dim, dropout=0.1, max_seq_len=10):
        super().__init__()
        self.embed = nn.Sequential(
            Layer(n_input, 128, bn=True, dropout=dropout),
            Layer(128, embed_dim, bn=True, dropout=dropout),
        )
        # Learned positions let attention distinguish ordering in short windows.
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: [batch, seq_len, n_input]
        batch_size, seq_len, n_input = x.shape
        x = x.reshape(batch_size * seq_len, n_input)
        x = self.embed(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = x + self.positional_encoding[:, :seq_len, :]
        return x


class TemporalAttention(nn.Module):
    """Self-attention over timesteps with residual normalization and pooling."""

    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Keep per-head weights for interpretability (no head averaging here).
        attn_out, attn_weights = self.mha(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.norm(x + attn_out)

        # Pool using the last query's attention over all keys, averaged by head.
        pooled_weights = attn_weights.mean(dim=1)[:, -1, :]
        pooled = (pooled_weights.unsqueeze(-1) * attended).sum(dim=1)

        return pooled, attn_weights


class Encoder(nn.Module):
    """VAE encoder from pooled temporal embedding to latent distribution."""

    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)

        in_dim = int(hparams.get("embed_dim", 64))
        layer_dims = [in_dim] + [int(s) for s in self.hparams.layer_sizes.split(",")]
        bn = bool(self.hparams.batch_norm)
        dropout = float(getattr(self.hparams, "dropout", 0.0))

        self.layers = nn.Sequential(
            *[
                Layer(layer_dims[i], layer_dims[i + 1], bn=bn, dropout=dropout)
                for i in range(len(layer_dims) - 1)
            ]
        )

        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        self.logvar = nn.Linear(layer_dims[-1], self.hparams.latent_dim)

        init.xavier_uniform_(self.mu.weight)
        init.xavier_uniform_(self.logvar.weight)

    def forward(self, x):
        h = self.layers(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_, h


class Decoder(nn.Module):
    """VAE decoder from latent vector to current-timestep sensor outputs."""

    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)

        hidden_dims = [self.hparams.latent_dim] + [
            int(s) for s in reversed(self.hparams.layer_sizes.split(","))
        ]

        if hasattr(self.hparams, "output_variables"):
            out_dim = len(self.hparams.output_variables)
        else:
            out_dim = int(getattr(self.hparams, "n_output", 13))

        bn = bool(self.hparams.batch_norm)
        dropout = float(getattr(self.hparams, "dropout", 0.0))

        self.layers = nn.Sequential(
            *[
                Layer(hidden_dims[i], hidden_dims[i + 1], bn=bn, dropout=dropout)
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)
        init.xavier_uniform_(self.reconstructed.weight)

    def forward(self, z):
        h = self.layers(z)
        recon = self.reconstructed(h)
        return recon


class TemporalCVAE(pl.LightningModule):
    """Temporal Conditional VAE with attention pooling over a short history."""

    def __init__(
        self,
        x_train_data=None,
        y_train=None,
        x_val_normal=None,
        y_val_normal=None,
        x_val_abnormal=None,
        y_val_abnormal=None,
        pids_train=None,
        **hparams,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "x_train_data",
                "y_train",
                "x_val_normal",
                "y_val_normal",
                "x_val_abnormal",
                "y_val_abnormal",
                "pids_train",
            ]
        )

        if not hparams:
            hparams = dict(self.hparams)

        n_input = self._n_input(hparams)
        past_history = int(hparams.get("past_history", 1))
        embed_dim = int(hparams.get("embed_dim", 64))
        dropout = float(hparams.get("dropout", 0.0))
        n_attention_heads = int(hparams.get("n_attention_heads", 2))

        self.temporal_embedding = TemporalEmbedding(
            n_input=n_input,
            embed_dim=embed_dim,
            dropout=dropout,
            max_seq_len=max(past_history, 10),
        )
        self.temporal_attention = TemporalAttention(
            embed_dim=embed_dim,
            n_heads=n_attention_heads,
            dropout=dropout,
        )
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.n_input = n_input
        self.past_history = past_history

        self.x_train = x_train_data
        self.y_train = y_train

        self.x_val_normal = x_val_normal
        self.y_val_normal = y_val_normal
        self.x_val_abnormal = x_val_abnormal
        self.y_val_abnormal = y_val_abnormal
        self._val_on_device = False

        self._sample_weights = self._build_sample_weights(pids_train)

    @staticmethod
    def _n_input(hparams):
        if "input_variables" in hparams:
            return len(hparams["input_variables"])
        return int(hparams.get("n_input", 7))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * float(self.hparams.stdev)
            return eps * std + mu
        return mu

    def forward(self, batch):
        x = batch.reshape(batch.shape[0], self.past_history, self.n_input)
        embedded = self.temporal_embedding(x)
        pooled, attn_weights = self.temporal_attention(embedded)
        mu, logvar, _ = self.encoder(pooled)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, attn_weights

    def reconstruct(self, x):
        """Return only reconstruction tensor for benchmark pipeline compatibility."""
        recon, _, _, _ = self.forward(x)
        return recon

    def get_attention_weights(self, x):
        """Expose temporal attention maps for interpretability/diagnostics."""
        _, _, _, attn_weights = self.forward(x)
        return attn_weights

    def loss_function(self, obs, recon, mu, logvar):
        loss_fn = str(getattr(self.hparams, "loss_fn", "mse")).lower()

        if loss_fn in {"huber", "smooth_l1", "smoothl1"}:
            recon_loss = F.smooth_l1_loss(recon, obs, reduction="mean")
        elif loss_fn == "mse":
            recon_loss = F.mse_loss(recon, obs, reduction="mean")
        else:
            raise ValueError(
                f"Unsupported loss_fn='{self.hparams.loss_fn}'. Expected one of: mse, smooth_l1, huber"
            )

        kld = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
        return recon_loss, kld

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon, mu, logvar, _ = self.forward(x)
        recon_loss, kld = self.loss_function(y, recon, mu, logvar)

        warmup_epochs = int(getattr(self.hparams, "kld_warmup_epochs", 30))
        warmup_scale = min(1.0, self.current_epoch / max(warmup_epochs, 1))
        beta = float(self.hparams.kld_beta) * warmup_scale

        loss = recon_loss + beta * kld

        self.log(
            "total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("kld", kld, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "kld_beta_eff",
            beta,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self):
        """Track separation quality between normal and abnormal validation sets."""
        if any(
            item is None
            for item in [
                self.x_val_normal,
                self.y_val_normal,
                self.x_val_abnormal,
                self.y_val_abnormal,
            ]
        ):
            return

        self.eval()
        try:
            device = next(self.parameters()).device

            if not self._val_on_device:
                self.x_val_normal = self.x_val_normal.to(device)
                self.y_val_normal = self.y_val_normal.to(device)
                self.x_val_abnormal = self.x_val_abnormal.to(device)
                self.y_val_abnormal = self.y_val_abnormal.to(device)
                self._val_on_device = True

            with torch.no_grad():
                recon_normal, _, _, _ = self.forward(self.x_val_normal)
                mse_normal = F.mse_loss(recon_normal, self.y_val_normal)

                recon_abnormal, _, _, _ = self.forward(self.x_val_abnormal)
                mse_abnormal = F.mse_loss(recon_abnormal, self.y_val_abnormal)

                ratio = mse_abnormal / (mse_normal + 1e-8)

            self.log("val_mse_normal", mse_normal, prog_bar=True, logger=True)
            self.log("val_mse_abnormal", mse_abnormal, prog_bar=True, logger=True)
            self.log("val_ratio", ratio, prog_bar=True, logger=True)
        finally:
            self.train()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-4,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=25,
            T_mult=1,
            eta_min=1e-9,
            last_epoch=-1,
        )
        return [opt], [sch]

    @staticmethod
    def _build_sample_weights(pids):
        """Balance pumps in mini-batches to avoid dominance by frequent IDs."""
        if pids is None:
            return None

        if torch.is_tensor(pids):
            pids_arr = pids.detach().cpu().numpy()
        else:
            pids_arr = np.asarray(pids)
        unique, counts = np.unique(pids_arr, return_counts=True)
        weight_map = {pid: 1.0 / cnt for pid, cnt in zip(unique, counts)}
        weights = np.array([weight_map[p] for p in pids_arr], dtype=np.float64)
        return torch.from_numpy(weights)

    @torch.no_grad()
    def anomaly_score_elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_kld: float = 0.05,
    ) -> torch.Tensor:
        """Compute per-sample ELBO anomaly score for E3 experiment.

        Combines the reconstruction loss with a weighted KLD term so that
        gradual thermal drift — which shifts the posterior q(z|x) away from
        the standard Gaussian prior without necessarily inflating the
        reconstruction error — can be detected.

        Score = recon_loss_i + lambda_kld * KLD_i,  for each sample i.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning input, shape [B, past_history * n_input].
            Same layout as the forward() batch input.
        y : torch.Tensor
            Ground-truth target outputs, shape [B, n_output].
        lambda_kld : float
            Weight applied to the KLD component. 0.0 reduces to pure MSE.

        Returns
        -------
        torch.Tensor
            Per-sample ELBO scores, shape [B].
        """
        was_training = self.training
        self.eval()
        try:
            # Encode through the temporal stack to obtain latent params.
            x_reshaped = x.reshape(x.shape[0], self.past_history, self.n_input)
            embedded = self.temporal_embedding(x_reshaped)
            pooled, _ = self.temporal_attention(embedded)
            mu, logvar, _ = self.encoder(pooled)

            # At eval time reparameterize returns mu deterministically; use it
            # for the reconstruction so the score is stable for a single pass.
            z = mu
            recon = self.decoder(z)

            # Per-sample reconstruction loss: mean over output dimensions.
            # Shape: [B, n_output] -> [B]
            recon_loss_per_sample = F.mse_loss(recon, y, reduction="none").mean(dim=1)

            # Per-sample KLD: sum over latent dimensions (analytic formula).
            # Clamp logvar to prevent exp() overflow on MPS/CPU.
            logvar_clamped = logvar.clamp(-30.0, 20.0)
            kld_per_sample = -0.5 * (
                1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()
            ).sum(dim=1)

            return recon_loss_per_sample + lambda_kld * kld_per_sample
        finally:
            if was_training:
                self.train()

    def train_dataloader(self):
        dataset = PumpDataset(self.x_train, self.y_train)
        on_cpu = not self.x_train.is_cuda

        if self._sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                pin_memory=on_cpu and should_pin_memory(),
                sampler=sampler,
            )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=on_cpu and should_pin_memory(),
            shuffle=True,
        )
