#!/usr/bin/env python3
"""Architecture validation for cond_reg_v2 — verifies shapes and forward pass."""
import sys
sys.path.insert(0, '<PATH_TO_PROJECT>')
import torch
import numpy as np

from cond_reg_v2.model.models import TemporalCVAE, Encoder, Decoder

# Parameters matching parameters.json
params = {
    "input_variables": ["Ambient temperature", "Main HTF Pump Speed", "Main HTF Pump Inlet Temperature",
                        "pump_id_1", "pump_id_2", "pump_id_3", "pump_id_4"],
    "output_variables": ["Main HTF Pump Current Consumption", "Main HTF Pump Flow",
                         "Main HTF Pump Outlet Pressure", "Main HTF Pump NDE Outboard bearing",
                         "Main HTF Pump NDE Inboard bearing", "Main HTF Pump DE bearing",
                         "Main HTF Pump Motor bearing Temp 1", "Main HTF Pump Motor bearing Temp 2",
                         "Main HTF Pump Motor U winding Temp 1", "Main HTF Pump Motor U winding Temp 2",
                         "Main HTF Pump Motor U winding Temp 3", "Main HTF Pump DE Side Bearing vibration",
                         "Main HTF Pump NDE Side Bearing vibration"],
    "past_history": 3,
    "latent_dim": 16,
    "layer_sizes": "256,128,64",
    "batch_norm": True,
    "dropout": 0.1,
    "stdev": 0.1,
    "kld_beta": 0.01,
    "kld_warmup_epochs": 30,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 150,
    "patience": 44,
    "loss_fn": "huber",
    "huber_delta": 1.0,
    "model": "temporal_cvae",
    "norm_method": "min-max",
}

n_input = len(params["input_variables"])  # 7
n_output = len(params["output_variables"])  # 13
past_history = params["past_history"]  # 3
batch_size = 8

print(f"Input: {n_input} features x {past_history} timesteps = {n_input * past_history} flat")
print(f"Output: {n_output} features")

# 1. Test Encoder with flattened input
flat_input_dim = past_history * n_input
params["flat_input_dim"] = flat_input_dim
enc = Encoder(**params)
x_flat = torch.randn(batch_size, flat_input_dim)
mu, logvar, h = enc(x_flat)
print(f"\n1. Encoder: {x_flat.shape} -> mu={mu.shape}, logvar={logvar.shape}")
assert mu.shape == (batch_size, params["latent_dim"]), f"mu shape mismatch: {mu.shape}"

# 2. Test Decoder
dec = Decoder(**params)
recon = dec(mu)
print(f"2. Decoder: {mu.shape} -> {recon.shape}")
assert recon.shape == (batch_size, n_output), f"recon shape mismatch: {recon.shape}"

# 3. Test full TemporalCVAE forward pass
y_dummy = torch.randn(batch_size, n_output)

# Create model with dummy data
model = TemporalCVAE(
    x_train_data=x_flat,
    y_train=y_dummy,
    x_val_normal=x_flat[:2],
    y_val_normal=y_dummy[:2],
    x_val_abnormal=x_flat[:2],
    y_val_abnormal=y_dummy[:2],
    **params
)

model.eval()
with torch.no_grad():
    out = model(x_flat)
    recon_out, mu_out, logvar_out = out
    print(f"\n3. Full forward: input={x_flat.shape} -> recon={recon_out.shape}, mu={mu_out.shape}")
    assert recon_out.shape == (batch_size, n_output), f"Full forward recon shape: {recon_out.shape}"

# 4. Test loss function
model.train()
recon_loss, kld = model.loss_function(y_dummy, recon_out, mu_out, logvar_out)
print(f"4. Loss: recon={recon_loss.item():.4f}, kld={kld.item():.4f}")

# 5. Test reconstruct method
model.eval()
with torch.no_grad():
    recon_only = model.reconstruct(x_flat)
    print(f"5. Reconstruct: {x_flat.shape} -> {recon_only.shape}")
    assert recon_only.shape == (batch_size, n_output)

# 6. Test attention weights extraction compatibility
with torch.no_grad():
    aw = model.get_attention_weights(x_flat)
    print(f"6. Attention weights: {aw}")
    assert aw is None

print("\n=== ALL ARCHITECTURE TESTS PASSED ===")
