from types import SimpleNamespace

import pytest
import torch

from model.comparison.alternative_models import LSTMVAE, TCNVAE


@pytest.fixture
def synthetic_setup():
    batch, T, n_in, n_out = 8, 24, 5, 5
    latent_dim = 16

    x_train = torch.randn(batch * 4, T * n_in)
    y_train = torch.randn(batch * 4, T * n_out)
    x_val_n = torch.randn(batch, T * n_in)
    y_val_n = torch.randn(batch, T * n_out)
    x_val_a = torch.randn(batch, T * n_in)
    y_val_a = torch.randn(batch, T * n_out)

    hparams = {
        "past_history": T,
        "latent_dim": latent_dim,
        "input_variables": [f"var_{i}" for i in range(n_in)],
        "output_variables": [f"var_{i}" for i in range(n_out)],
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": batch,
        "epochs": 10,
        "patience": 5,
        "kld_beta": 0.005,
        "kld_warmup_epochs": 50,
        "stdev": 0.1,
    }
    return {
        "batch": batch,
        "T": T,
        "n_out": n_out,
        "latent_dim": latent_dim,
        "x_train": x_train,
        "y_train": y_train,
        "x_val_n": x_val_n,
        "y_val_n": y_val_n,
        "x_val_a": x_val_a,
        "y_val_a": y_val_a,
        "hparams": hparams,
    }


@pytest.fixture(params=[LSTMVAE, TCNVAE], ids=["lstm_vae", "tcn_vae"])
def model(request, synthetic_setup):
    model_cls = request.param
    m = model_cls(
        synthetic_setup["x_train"],
        synthetic_setup["y_train"],
        synthetic_setup["x_val_n"],
        synthetic_setup["y_val_n"],
        synthetic_setup["x_val_a"],
        synthetic_setup["y_val_a"],
        **synthetic_setup["hparams"],
    )
    return m


def _set_current_epoch(model, epoch: int):
    model._trainer = SimpleNamespace(current_epoch=epoch)


def test_forward_pass_shapes(model, synthetic_setup):
    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    n_out = synthetic_setup["n_out"]
    latent_dim = synthetic_setup["latent_dim"]

    x = torch.randn(batch, T * len(model.hparams.input_variables))
    recon, mu, logvar = model.forward(x)

    assert recon.shape == (batch, T * n_out)
    assert mu.shape == (batch, latent_dim)
    assert logvar.shape == (batch, latent_dim)


def test_reparameterization_stochasticity_train_vs_eval(model, synthetic_setup):
    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    x = torch.randn(batch, T * len(model.hparams.input_variables))

    model.train()
    _, mu_t1, logvar_t1 = model.forward(x)
    _, mu_t2, logvar_t2 = model.forward(x)
    z_t1 = model._reparameterize(mu_t1, logvar_t1)
    z_t2 = model._reparameterize(mu_t2, logvar_t2)

    assert not torch.allclose(z_t1, mu_t1)
    assert not torch.allclose(z_t2, mu_t2)
    assert not torch.allclose(z_t1 - mu_t1, z_t2 - mu_t2)

    model.eval()
    with torch.no_grad():
        _, mu_e, logvar_e = model.forward(x)
        z_e = model._reparameterize(mu_e, logvar_e)
    assert torch.allclose(z_e, mu_e)


def test_kld_warmup_beta_behavior(model, synthetic_setup):
    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    n_out = synthetic_setup["n_out"]

    x = torch.randn(batch, T * len(model.hparams.input_variables))
    y = torch.randn(batch, T * n_out)

    recon = torch.randn_like(y)
    mu = torch.randn(batch, model.hparams.latent_dim)
    logvar = torch.randn(batch, model.hparams.latent_dim)

    model.forward = lambda _: (recon, mu, logvar)

    recon_loss = torch.nn.functional.mse_loss(recon, y, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    _set_current_epoch(model, 0)
    loss_e0, _ = model.compute_loss(x, y)
    assert torch.allclose(loss_e0, recon_loss, atol=1e-6)

    _set_current_epoch(model, model.hparams.kld_warmup_epochs)
    loss_ew, _ = model.compute_loss(x, y)
    expected = recon_loss + model.hparams.kld_beta * kld
    assert torch.allclose(loss_ew, expected, atol=1e-6)


def test_compute_loss_contract(model, synthetic_setup):
    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    n_out = synthetic_setup["n_out"]

    x = torch.randn(batch, T * len(model.hparams.input_variables))
    y = torch.randn(batch, T * n_out)

    loss, logs = model.compute_loss(x, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert isinstance(logs, dict)
    assert "recon_loss" in logs
    assert "kld_loss" in logs


def test_reconstruct_output_shape(model, synthetic_setup):
    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    n_out = synthetic_setup["n_out"]

    x = torch.randn(batch, T * len(model.hparams.input_variables))
    recon = model.reconstruct(x)

    assert isinstance(recon, torch.Tensor)
    assert recon.ndim == 2
    assert recon.shape == (batch, T * n_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("model_cls", [LSTMVAE, TCNVAE], ids=["lstm_vae", "tcn_vae"])
def test_reconstruct_output_shape_on_cuda(model_cls, synthetic_setup):
    device = torch.device("cuda")
    hparams = synthetic_setup["hparams"]

    model = model_cls(
        synthetic_setup["x_train"].to(device),
        synthetic_setup["y_train"].to(device),
        synthetic_setup["x_val_n"].to(device),
        synthetic_setup["y_val_n"].to(device),
        synthetic_setup["x_val_a"].to(device),
        synthetic_setup["y_val_a"].to(device),
        **hparams,
    ).to(device)

    batch = synthetic_setup["batch"]
    T = synthetic_setup["T"]
    n_out = synthetic_setup["n_out"]
    x = torch.randn(batch, T * len(hparams["input_variables"]), device=device)

    recon = model.reconstruct(x)
    assert recon.ndim == 2
    assert recon.shape == (batch, T * n_out)
