"""Alternative neural network models for anomaly detection benchmarking against VAE.

Models implemented:
    - StandardAE:       Standard autoencoder (no variational component)
    - SparseAE:         Autoencoder with L1 sparsity penalty on latent space
    - DenoisingAE:      Denoising autoencoder (learns from corrupted inputs)
    - LSTMAutoencoder:  LSTM-based sequence autoencoder (temporal awareness)
    - CNNAutoencoder:   1D convolutional autoencoder (local temporal patterns)
    - TransformerAE:    Transformer-based autoencoder (self-attention)

All models share the same constructor signature and evaluation loop as the
existing VAE so they can be benchmarked side-by-side with identical data
splits, metrics, and training infrastructure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..models import PumpDataset, Layer
from ..device import should_pin_memory


# ============================================================================
# BASE CLASS — shared by all reconstruction-based models
# ============================================================================

class BaseReconstructionModel(pl.LightningModule):
    """Base class for reconstruction-based anomaly detection models.

    Provides:
        - Data storage and dataloader creation
        - Validation loop tracking MAE ratio (abnormal / normal)
        - AdamW + CosineAnnealingWarmRestarts optimizer configuration

    Subclasses must implement:
        - forward(x)          → reconstruction (or tuple whose first element is recon)
        - compute_loss(x, y)  → (scalar loss, dict of logged metrics)
    """

    def __init__(self, x_train_data=None, y_train=None, x_val_normal=None,
                 y_val_normal=None, x_val_abnormal=None, y_val_abnormal=None,
                 pids_train=None, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'x_train_data', 'y_train', 'x_val_normal', 'y_val_normal',
            'x_val_abnormal', 'y_val_abnormal', 'pids_train',
        ])

        # Common dimensions computed from parameter lists
        self.n_input = len(self.hparams.input_variables)
        self.n_output = len(self.hparams.output_variables)
        self.in_dim = self.n_input * self.hparams.past_history
        self.out_dim = self.n_output * self.hparams.past_history

        # Store datasets
        self.x_train = x_train_data
        self.y_train = y_train
        self.x_val_normal = x_val_normal
        self.y_val_normal = y_val_normal
        self.x_val_abnormal = x_val_abnormal
        self.y_val_abnormal = y_val_abnormal
        self._val_on_device = False

        # Build per-sample weights for pump-balanced sampling
        self._sample_weights = self._build_sample_weights(pids_train)

    @staticmethod
    def _build_sample_weights(pids):
        """Compute per-sample weights so every pump is equally represented."""
        if pids is None:
            return None
        pids_arr = np.asarray(pids)
        unique, counts = np.unique(pids_arr, return_counts=True)
        weight_map = {pid: 1.0 / cnt for pid, cnt in zip(unique, counts)}
        weights = np.array([weight_map[p] for p in pids_arr], dtype=np.float64)
        return torch.from_numpy(weights)

    # ------------------------------------------------------------------
    # Reconstruction helper
    # ------------------------------------------------------------------
    def reconstruct(self, x):
        """Return only the reconstruction tensor from forward()."""
        output = self.forward(x)
        if isinstance(output, tuple):
            return output[0]
        return output

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, logs = self.compute_loss(x, y)
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in logs.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """Compute validation MSE ratio (MSE_abnormal / MSE_normal) every epoch."""
        self.eval()
        try:
            device = next(self.parameters()).device

            if not self._val_on_device:
                self.x_val_normal = self.x_val_normal.to(device)
                self.x_val_abnormal = self.x_val_abnormal.to(device)
                self.y_val_normal = self.y_val_normal.to(device)
                self.y_val_abnormal = self.y_val_abnormal.to(device)
                self._val_on_device = True

            with torch.no_grad():
                recon_normal = self.reconstruct(self.x_val_normal)
                mse_normal = F.mse_loss(recon_normal, self.y_val_normal)

                recon_abnormal = self.reconstruct(self.x_val_abnormal)
                mse_abnormal = F.mse_loss(recon_abnormal, self.y_val_abnormal)

                ratio = mse_abnormal / (mse_normal + 1e-8)

            self.log('val_mse_normal', mse_normal, prog_bar=True, logger=True)
            self.log('val_mse_abnormal', mse_abnormal, prog_bar=True, logger=True)
            self.log('val_ratio', ratio, prog_bar=True, logger=True)
        finally:
            self.train()

    # ------------------------------------------------------------------
    # Optimizer — same as VAE for fair comparison
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay, eps=1e-4,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1,
        )
        return [opt], [sch]

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
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
                dataset, batch_size=self.hparams.batch_size,
                pin_memory=on_cpu and should_pin_memory(),
                sampler=sampler,
            )

        return DataLoader(
            dataset, batch_size=self.hparams.batch_size,
            pin_memory=on_cpu and should_pin_memory(), shuffle=True,
        )

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    def compute_loss(self, x, y):
        """Return (loss, logs_dict). Must be implemented by subclasses."""
        raise NotImplementedError


# ============================================================================
# 1. STANDARD AUTOENCODER (AE)
# ============================================================================

class StandardAE(BaseReconstructionModel):
    """Standard autoencoder — no variational component.

    Same encoder/decoder architecture as VAE but without mu/logvar split,
    reparameterization trick, or KLD regularization.

    Architecture:
        Encoder: Input → [512 → 256 → 128] → Latent(32)
        Decoder: Latent(32) → [128 → 256 → 512] → Output
    Loss: MSE (reconstruction only)
    """

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        # Encoder
        layer_dims = [self.in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.encoder = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)]
        )
        self.to_latent = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        init.xavier_uniform_(self.to_latent.weight)

        # Decoder
        dec_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        self.decoder = nn.Sequential(
            *[Layer(dec_dims[i], dec_dims[i + 1], bn) for i in range(len(dec_dims) - 1)]
        )
        self.output_layer = nn.Linear(dec_dims[-1], self.out_dim)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        h = self.encoder(x)
        z = self.to_latent(h)
        h_dec = self.decoder(z)
        recon = self.output_layer(h_dec)
        return recon

    def compute_loss(self, x, y):
        recon = self.forward(x)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}


# ============================================================================
# 2. SPARSE AUTOENCODER (SAE)
# ============================================================================

class SparseAE(BaseReconstructionModel):
    """Autoencoder with L1 sparsity penalty on latent activations.

    Encourages the latent representation to be sparse (mostly zeros),
    which can improve anomaly detection by making the model more selective
    about which features are activated for normal data.

    Loss: MSE + λ · ||z||₁
    """

    SPARSITY_LAMBDA = 1e-3  # default sparsity coefficient

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        hparams.setdefault('sparsity_lambda', self.SPARSITY_LAMBDA)
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        # Encoder (same architecture as StandardAE)
        layer_dims = [self.in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.encoder = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)]
        )
        self.to_latent = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        init.xavier_uniform_(self.to_latent.weight)

        # Decoder
        dec_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        self.decoder = nn.Sequential(
            *[Layer(dec_dims[i], dec_dims[i + 1], bn) for i in range(len(dec_dims) - 1)]
        )
        self.output_layer = nn.Linear(dec_dims[-1], self.out_dim)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        h = self.encoder(x)
        z = self.to_latent(h)
        h_dec = self.decoder(z)
        recon = self.output_layer(h_dec)
        return recon, z  # return latent for sparsity penalty

    def compute_loss(self, x, y):
        recon, z = self.forward(x)
        recon_loss = F.mse_loss(recon, y, reduction='mean')
        sparsity = z.abs().mean()
        loss = recon_loss + self.hparams.sparsity_lambda * sparsity
        return loss, {'recon_loss': recon_loss, 'sparsity': sparsity}


# ============================================================================
# 3. DENOISING AUTOENCODER (DAE)
# ============================================================================

class DenoisingAE(BaseReconstructionModel):
    """Denoising autoencoder — learns robust features by reconstructing
    clean data from corrupted inputs.

    During training: adds Gaussian noise to input, reconstructs clean target.
    During inference: no noise added (uses clean input directly).

    Loss: MSE on clean reconstruction
    """

    NOISE_FACTOR = 0.1  # default noise standard deviation

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        hparams.setdefault('noise_factor', self.NOISE_FACTOR)
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        # Same architecture as StandardAE
        layer_dims = [self.in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.encoder = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)]
        )
        self.to_latent = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        init.xavier_uniform_(self.to_latent.weight)

        dec_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        self.decoder = nn.Sequential(
            *[Layer(dec_dims[i], dec_dims[i + 1], bn) for i in range(len(dec_dims) - 1)]
        )
        self.output_layer = nn.Linear(dec_dims[-1], self.out_dim)
        init.xavier_uniform_(self.output_layer.weight)

    def _encode_decode(self, x):
        """Shared encoder-decoder pass."""
        h = self.encoder(x)
        z = self.to_latent(h)
        h_dec = self.decoder(z)
        return self.output_layer(h_dec)

    def forward(self, x):
        # During inference: no noise
        return self._encode_decode(x)

    def compute_loss(self, x, y):
        # During training: corrupt input with Gaussian noise
        if self.training:
            noise = torch.randn_like(x) * self.hparams.noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x
        recon = self._encode_decode(x_noisy)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}


# ============================================================================
# 4. LSTM AUTOENCODER
# ============================================================================

class LSTMAutoencoder(BaseReconstructionModel):
    """LSTM-based sequence autoencoder for temporal anomaly detection.

    Processes temporal patterns natively using recurrent connections.
    The encoder LSTM compresses the full sequence into a fixed-size latent
    vector; the decoder LSTM reconstructs the sequence step-by-step.

    Architecture:
        Encoder: LSTM(n_input, 128, 2 layers) → last_hidden → Linear → latent
        Decoder: Linear(latent, 128) → repeat T → LSTM(128, 128, 2 layers) → Linear → n_output
                 → flatten to (batch, T × n_output)
    """

    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.1

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        hparams.setdefault('lstm_hidden', self.LSTM_HIDDEN)
        hparams.setdefault('lstm_layers', self.LSTM_LAYERS)
        hparams.setdefault('lstm_dropout', self.LSTM_DROPOUT)
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        hidden = self.hparams.lstm_hidden
        n_layers = self.hparams.lstm_layers
        dropout = self.hparams.lstm_dropout if n_layers > 1 else 0.0

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.n_input,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.to_latent = nn.Linear(hidden, self.hparams.latent_dim)

        # Decoder
        self.from_latent = nn.Linear(self.hparams.latent_dim, hidden)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden, self.n_output)

        # Initialize linear layers
        init.xavier_uniform_(self.to_latent.weight)
        init.xavier_uniform_(self.from_latent.weight)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        batch_size = x.size(0)
        T = self.hparams.past_history

        # Reshape flat → sequence: (batch, T, n_input)
        x_seq = x.view(batch_size, T, self.n_input)

        # Encode: take last layer's final hidden state
        _, (h_n, _) = self.encoder_lstm(x_seq)       # h_n: (n_layers, batch, hidden)
        latent_input = h_n[-1]                        # (batch, hidden)
        z = self.to_latent(latent_input)              # (batch, latent_dim)

        # Decode: expand latent → repeat T times → LSTM → project
        h_dec = F.leaky_relu(self.from_latent(z), 0.1)    # (batch, hidden)
        h_dec_seq = h_dec.unsqueeze(1).repeat(1, T, 1)    # (batch, T, hidden)
        dec_out, _ = self.decoder_lstm(h_dec_seq)          # (batch, T, hidden)
        recon_seq = self.output_layer(dec_out)             # (batch, T, n_output)

        # Flatten back: (batch, T × n_output)
        recon = recon_seq.reshape(batch_size, -1)
        return recon

    def compute_loss(self, x, y):
        recon = self.forward(x)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}


# ============================================================================
# 5. 1D-CNN AUTOENCODER
# ============================================================================

class CNNAutoencoder(BaseReconstructionModel):
    """1D Convolutional autoencoder for temporal anomaly detection.

    Captures local temporal patterns through convolution filters.
    Uses Conv1d for encoding and ConvTranspose1d / interpolation for decoding.

    Architecture:
        Encoder: Conv1d(n_input→64→128→256) + BN + LeakyReLU → AdaptivePool → Linear → latent
        Decoder: Linear → reshape → Upsample → ConvT1d(256→128→64) → Conv1d → n_output → flatten
    """

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        T = self.hparams.past_history

        # Encoder: Conv1d expects (batch, channels, length) = (batch, n_input, T)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(self.n_input, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (batch, 256, 1)
        )
        self.encoder_fc = nn.Linear(256, self.hparams.latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(self.hparams.latent_dim, 256)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, self.n_output, kernel_size=3, padding=1),
        )

        self._T = T

        init.xavier_uniform_(self.encoder_fc.weight)
        init.xavier_uniform_(self.decoder_fc.weight)

    def forward(self, x):
        batch_size = x.size(0)
        T = self._T

        # Reshape flat → Conv1d format: (batch, n_input, T)
        x_conv = x.view(batch_size, T, self.n_input).permute(0, 2, 1)

        # Encode
        h = self.encoder_conv(x_conv)       # (batch, 256, 1)
        h = h.squeeze(-1)                   # (batch, 256)
        z = self.encoder_fc(h)              # (batch, latent_dim)

        # Decode
        h_dec = F.leaky_relu(self.decoder_fc(z), 0.1)    # (batch, 256)
        h_dec = h_dec.unsqueeze(-1)                       # (batch, 256, 1)
        h_dec = F.interpolate(h_dec, size=T, mode='nearest')  # (batch, 256, T)
        recon_conv = self.decoder_conv(h_dec)             # (batch, n_output, T)

        # Flatten: (batch, n_output, T) → (batch, T × n_output)
        recon = recon_conv.permute(0, 2, 1).reshape(batch_size, -1)
        return recon

    def compute_loss(self, x, y):
        recon = self.forward(x)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}


# ============================================================================
# POSITIONAL ENCODING (for Transformer)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# ============================================================================
# 6. TRANSFORMER AUTOENCODER
# ============================================================================

class TransformerAE(BaseReconstructionModel):
    """Transformer-based autoencoder with multi-head self-attention.

    Captures long-range temporal dependencies that local models (CNN, LSTM)
    may miss.  Uses TransformerEncoder layers for both encoding and decoding.

    Architecture:
        Encoder: Linear(n_input → d_model) + PosEnc → TransformerEncoder × 2
                 → mean-pool over time → Linear → latent
        Decoder: Linear(latent → d_model) → repeat T + PosEnc
                 → TransformerEncoder × 2 → Linear(d_model → n_output) → flatten
    """

    D_MODEL = 64
    N_HEADS = 4
    N_TRANSFORMER_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        hparams.setdefault('d_model', self.D_MODEL)
        hparams.setdefault('n_heads', self.N_HEADS)
        hparams.setdefault('n_transformer_layers', self.N_TRANSFORMER_LAYERS)
        hparams.setdefault('transformer_dropout', self.TRANSFORMER_DROPOUT)
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        d_model = self.hparams.d_model
        n_heads = self.hparams.n_heads
        n_layers = self.hparams.n_transformer_layers
        dropout = self.hparams.transformer_dropout
        T = self.hparams.past_history

        # Encoder
        self.input_projection = nn.Linear(self.n_input, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Linear(d_model, self.hparams.latent_dim)

        # Decoder
        self.from_latent = nn.Linear(self.hparams.latent_dim, d_model)
        self.pos_enc_dec = PositionalEncoding(d_model, max_len=T)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, self.n_output)

        self._T = T

        # Initialize
        init.xavier_uniform_(self.input_projection.weight)
        init.xavier_uniform_(self.to_latent.weight)
        init.xavier_uniform_(self.from_latent.weight)
        init.xavier_uniform_(self.output_projection.weight)

    def forward(self, x):
        batch_size = x.size(0)
        T = self._T

        # Reshape flat → sequence: (batch, T, n_input)
        x_seq = x.view(batch_size, T, self.n_input)

        # Encode
        h = self.input_projection(x_seq)          # (batch, T, d_model)
        h = self.pos_enc(h)
        h = self.transformer_encoder(h)           # (batch, T, d_model)
        h_pooled = h.mean(dim=1)                  # (batch, d_model) — mean pool over time
        z = self.to_latent(h_pooled)              # (batch, latent_dim)

        # Decode
        h_dec = F.leaky_relu(self.from_latent(z), 0.1)   # (batch, d_model)
        h_dec = h_dec.unsqueeze(1).repeat(1, T, 1)        # (batch, T, d_model)
        h_dec = self.pos_enc_dec(h_dec)
        h_dec = self.transformer_decoder(h_dec)           # (batch, T, d_model)
        recon_seq = self.output_projection(h_dec)         # (batch, T, n_output)

        # Flatten: (batch, T × n_output)
        recon = recon_seq.reshape(batch_size, -1)
        return recon

    def compute_loss(self, x, y):
        recon = self.forward(x)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}


# ============================================================================
# 7. USAD — UnSupervised Anomaly Detection (Audibert et al. 2020)
# ============================================================================

class USADModel(BaseReconstructionModel):
    """USAD — UnSupervised Anomaly Detection with dual autoencoders.

    Uses a shared encoder and two competing decoders:
        - Decoder1 (AE1): trained to reconstruct the original input
        - Decoder2 (AE2): trained to distinguish real data from AE1 reconstructions

    This adversarial setup forces AE1 to produce increasingly faithful
    reconstructions while AE2 becomes better at detecting subtle anomalies.

    Architecture:
        Encoder:  Linear(in → in/2 → in/4 → latent) + ReLU
        Decoder1: Linear(latent → in/4 → in/2 → in) + ReLU + Sigmoid
        Decoder2: Linear(latent → in/4 → in/2 → in) + ReLU + Sigmoid

    Training (manual optimization — 2 separate optimizers):
        loss_AE1 = (1/n)||W - W1||² + (1 - 1/n)||W - W3||²
        loss_AE2 = (1/n)||W - W2||² - (1 - 1/n)||W - W3||²
        where W1 = D1(E(W)), W2 = D2(E(W)), W3 = D2(E(W1)), n = epoch

    Anomaly score: α·||W - W1||² + β·||W - W3||²  (α=β=0.5 default)

    Reference:
        Audibert, J., et al. (2020). "USAD: UnSupervised Anomaly Detection
        on Multivariate Time Series." KDD 2020.
    """

    USAD_ALPHA = 0.5
    USAD_BETA = 0.5

    def __init__(self, x_train_data, y_train, x_val_normal, y_val_normal,
                 x_val_abnormal, y_val_abnormal, **hparams):
        hparams.setdefault('usad_alpha', self.USAD_ALPHA)
        hparams.setdefault('usad_beta', self.USAD_BETA)
        super().__init__(x_train_data, y_train, x_val_normal, y_val_normal,
                         x_val_abnormal, y_val_abnormal, **hparams)

        # USAD adapts the dual-decoder paradigm to our input→output mapping:
        #   Encoder compresses the full input (in_dim, includes pump_id dummies)
        #   Decoders reconstruct the output variables (out_dim, sensor data only)
        w_in = self.in_dim     # encoder input size
        w_out = self.out_dim   # decoder output size (reconstruction target)
        z_size = self.hparams.latent_dim

        # Shared Encoder — compresses full input
        self.usad_encoder = nn.Sequential(
            nn.Linear(w_in, w_in // 2),
            nn.ReLU(True),
            nn.Linear(w_in // 2, w_in // 4),
            nn.ReLU(True),
            nn.Linear(w_in // 4, z_size),
            nn.ReLU(True),
        )

        # Decoder 1 (reconstructor) — outputs sensor variables
        self.decoder1 = nn.Sequential(
            nn.Linear(z_size, w_in // 4),
            nn.ReLU(True),
            nn.Linear(w_in // 4, w_in // 2),
            nn.ReLU(True),
            nn.Linear(w_in // 2, w_out),
            nn.Sigmoid(),
        )

        # Decoder 2 (discriminator/detector) — outputs sensor variables
        self.decoder2 = nn.Sequential(
            nn.Linear(z_size, w_in // 4),
            nn.ReLU(True),
            nn.Linear(w_in // 4, w_in // 2),
            nn.ReLU(True),
            nn.Linear(w_in // 2, w_out),
            nn.Sigmoid(),
        )

        # USAD needs manual optimization (two separate optimizer steps)
        self.automatic_optimization = False

        # Projection layer for re-encoding: maps decoder output (out_dim)
        # back to encoder input space (in_dim) for the W3 = D2(E(D1(E(x)))) path
        if w_in != w_out:
            self.reproject = nn.Linear(w_out, w_in)
        else:
            self.reproject = nn.Identity()

    def forward(self, x):
        """Standard reconstruction via Decoder1 (used for evaluation/scoring)."""
        z = self.usad_encoder(x)
        w1 = self.decoder1(z)
        return w1

    def reconstruct(self, x):
        """Return Decoder1 reconstruction (anomaly-scoring compatible)."""
        return self.forward(x)

    def _usad_forward(self, x):
        """Full USAD forward: returns W1, W2, W3 for loss computation."""
        z = self.usad_encoder(x)
        w1 = self.decoder1(z)                        # AE1 reconstruction (out_dim)
        w2 = self.decoder2(z)                         # AE2 reconstruction (out_dim)
        # Re-encode W1: project back to in_dim, then encode + decode through AE2
        w1_proj = self.reproject(w1)                  # (batch, in_dim)
        z_w1 = self.usad_encoder(w1_proj)
        w3 = self.decoder2(z_w1)                      # AE2(AE1(x))
        return w1, w2, w3

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt_ae1, opt_ae2 = self.optimizers()

        # epoch number (1-indexed for USAD loss weighting)
        n = self.current_epoch + 1

        # --- Train AE1 (encoder + decoder1) ---
        w1, w2, w3 = self._usad_forward(x)
        loss_ae1 = (1 / n) * torch.mean((y - w1) ** 2) + \
                   (1 - 1 / n) * torch.mean((y - w3) ** 2)

        opt_ae1.zero_grad()
        self.manual_backward(loss_ae1)
        opt_ae1.step()

        # --- Train AE2 (encoder + decoder2) ---
        w1, w2, w3 = self._usad_forward(x)
        loss_ae2 = (1 / n) * torch.mean((y - w2) ** 2) - \
                   (1 - 1 / n) * torch.mean((y - w3) ** 2)

        opt_ae2.zero_grad()
        self.manual_backward(loss_ae2)
        opt_ae2.step()

        # Log combined loss for monitoring
        total_loss = loss_ae1 + loss_ae2
        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_ae1', loss_ae1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_ae2', loss_ae2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def compute_loss(self, x, y):
        """Not used directly (manual optimization), but required by base class."""
        recon = self.forward(x)
        loss = F.mse_loss(recon, y, reduction='mean')
        return loss, {'recon_loss': loss}

    def configure_optimizers(self):
        """Two separate optimizers: one for AE1 path, one for AE2 path.
        
        The reproject layer and encoder are shared, so they appear in both.
        """
        shared_params = list(self.usad_encoder.parameters())
        if isinstance(self.reproject, nn.Linear):
            shared_params += list(self.reproject.parameters())

        opt_ae1 = torch.optim.AdamW(
            shared_params + list(self.decoder1.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-4,
        )
        opt_ae2 = torch.optim.AdamW(
            shared_params + list(self.decoder2.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=1e-4,
        )
        return [opt_ae1, opt_ae2]
