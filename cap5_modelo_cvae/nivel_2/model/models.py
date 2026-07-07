# Models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from argparse import Namespace
from .device import should_pin_memory


class PumpDataset(Dataset):
    """
    Dataset object in order for PyTorch to handle the dataframe generated in preprocessing.
    """
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data    # Store the inputs (Speed, Temp, ID)
        self.y_data = y_data    # Store the targets (vibration, winding temperature)

    def __getitem__(self, idx): 
        # Devolvemos un par de (entrada, etiqueta)
        return self.x_data[idx], self.y_data[idx] # Returns a pair: (Input, Target)
    
    def __len__(self):
        return self.x_data.shape[0] # tells the model how many total samples exist



class Layer(nn.Module):
    '''
    A single fully connected layer with optional batch normalization and activation.
    '''
    def __init__(self, in_dim, out_dim, bn = True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)] # Linear layer
        if bn: layers.append(nn.BatchNorm1d(out_dim)) # optional batch normalization
        layers.append(nn.LeakyReLU(0.1, inplace=True)) # activation - LeakyReLU (slope 0.1)
        self.block = nn.Sequential(*layers) # pack the 3 steps in a sequential block

        init.xavier_uniform_(self.block[0].weight) # xavier uniform initialization of weights (avoid exploding/vanishing)
                                                   # intialization with _ (uniform_ instead of uniform): overwrite data right there (faster, efficient)

    def forward(self, x): # Forward pass (Linear -> BN (optional) -> LeakyReLU)
        return self.block(x) 

class Encoder(nn.Module):
    '''
    The encoder part of our VAE. Takes a data sample and returns the mean and the log-variance of the 
    latent vector's distribution.
    '''
    def __init__(self, **hparams): 
        super().__init__()
        self.hparams = Namespace(**hparams) # convert dictionary params to object (params.lr) 
        # '**' break the dictionary hparams into kwargs (named arguments) -> {"latent_dim":16} into lantent_dim = 16

        # Calculate input size (number variables) * (time steps)
        in_dim = len(self.hparams.input_variables) * self.hparams.past_history

        # define the layers structure (parameters.json - layer_sizes) : [input] -> [512, 256, 128]
        layer_dims = [in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm # define batch normalization

        # create the main NN layers using the 'Layer' class defined before
        self.layers = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)],
        ) # '*' -> unpacking the comprehension list of layers [L1, L2, L3] into separate arguments: L1, L2, L3

        # VAE split: last layer splits into 2 parallel outputs
        # 1. 'mu': Mean (best guess of the latent state)
        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        # 2. 'logvar': Log Variance (uncertainty/noise of the state)
        self.logvar = nn.Linear(layer_dims[-1], self.hparams.latent_dim)

        # Initialize weights
        init.xavier_uniform_(self.mu.weight)
        init.xavier_uniform_(self.logvar.weight)
        
    
    def forward(self, x):
        h = self.layers(x)  # Pass input through 512->256->128
        # Produce the two statistical components
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_, h
    
class Decoder(nn.Module):
    '''
    The decoder part of our VAE. Takes a latent vector (sampled from the distribution learned by the 
    encoder) and converts it back to a reconstructed data sample.
    '''
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)

        # VAE decoder input is the latent vector
        hidden_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]

        out_dim = len(self.hparams.output_variables) * self.hparams.past_history    # Full VAE: output_dim must match input_dim
        bn = self.hparams.batch_norm

        # Build the layers
        self.layers = nn.Sequential(
            *[Layer(hidden_dims[i], hidden_dims[i + 1], bn) for i in range(len(hidden_dims) - 1)],
        )
        # Final output layer (Linear) to map back to original units (normalized 0-1)
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)

        init.xavier_uniform_(self.reconstructed.weight)
        
    def forward(self, z):   # z: latent vector (compressed numerical summary of the pump's data - 16 abstract numbers) - generated in VAE.forward 
        h = self.layers(z)      
        recon = self.reconstructed(h)
        return recon
    
class VAE(pl.LightningModule):
    """
    Connects Encoder and Decoder and defines the reparameterization
    """
    def __init__(self, x_train_data=None, y_train=None, x_val_normal=None, y_val_normal=None, x_val_abnormal=None, y_val_abnormal=None, pids_train=None, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'x_train_data', 'y_train', 'x_val_normal', 'y_val_normal',
            'x_val_abnormal', 'y_val_abnormal', 'pids_train',
        ]) # saves all params to a file (reproducibility)
        self.encoder = Encoder(**hparams) if hparams else Encoder(**dict(self.hparams))
        self.decoder = Decoder(**hparams) if hparams else Decoder(**dict(self.hparams))
        # Store datasets
        self.x_train = x_train_data
        self.y_train = y_train
        self.x_val_normal = x_val_normal
        self.y_val_normal = y_val_normal
        self.x_val_abnormal = x_val_abnormal
        self.y_val_abnormal = y_val_abnormal
        self._val_on_device = False  # flag: validation tensors moved to GPU yet?

        # Build per-sample weights for pump-balanced sampling
        self._sample_weights = self._build_sample_weights(pids_train)
        
    def reparameterize(self, mu, logvar):
        '''
        The reparameterization trick allows us to backpropagate through the encoder.
        Instead of passing 'mu' directly, sample a point from the Normal Distribution N(mu, std).
        This forces the model to learn a smooth "cloud" of data, not just single points.
        '''
        if self.training:
            std = torch.exp(0.5 * logvar) # convert log-variance to standard deviation
            eps = torch.randn_like(std) * self.hparams.stdev    # generate random noise
            return eps * std + mu # shift and scale: Z = Mean + (Noise * Std)
        else:
            return mu   # during testing, no randomness -> the selected must be the best guess
        
    def forward(self, batch):
        x = batch  
        mu, logvar, h = self.encoder(x) # encode
        z = self.reparameterize(mu, logvar) # add noise (if training)
        recon = self.decoder(z) # decode/reconstruct
        return recon, mu, logvar, x

    def reconstruct(self, x):
        """Return only the reconstruction tensor (for benchmark compatibility)."""
        recon, _, _, _ = self.forward(x)
        return recon

    def loss_function(self, obs, recon, mu, logvar): # reconstruction loss
        # L2 loss (MSE) — penalises large errors more heavily
        recon_loss = F.mse_loss(recon, obs, reduction='mean')
        # KL Divergence (KLD): sees if the latent space is organized like a bell curve
        # this regularization term prevents overfitting
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld
                                 
    def training_step(self, batch, batch_idx):
        x, y = batch
        recon, mu, logvar, x = self.forward(x) # run the model
        recon_loss, kld = self.loss_function(y, recon, mu, logvar)

        # KLD warmup: linearly ramp beta over the first N epochs to prevent
        # posterior collapse (latent space ignored when beta is too high early on)
        warmup_epochs = getattr(self.hparams, 'kld_warmup_epochs', 30)
        beta = self.hparams.kld_beta * min(1.0, self.current_epoch / max(warmup_epochs, 1))
        loss = recon_loss + beta * kld # combine errors: accuracy + (beta * Regularization)

        # Registro de pérdidas - log metrics for graphs (WandB / Tensorboard)
        self.log('total_loss', loss.mean(dim=0), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_loss', recon_loss.mean(dim=0), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld', kld.mean(dim=0), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """Compute validation ratio (MSE_abnormal / MSE_normal) every epoch."""

        self.eval()
        try:
            device = next(self.parameters()).device

            # Move validation tensors to GPU once (first call), keep them there
            if not self._val_on_device:
                self.x_val_normal = self.x_val_normal.to(device)
                self.x_val_abnormal = self.x_val_abnormal.to(device)
                self.y_val_normal = self.y_val_normal.to(device)
                self.y_val_abnormal = self.y_val_abnormal.to(device)
                self._val_on_device = True

            with torch.no_grad():
                # Normal validation MSE
                recon_normal, _, _, _ = self.forward(self.x_val_normal)
                mse_normal = F.mse_loss(recon_normal, self.y_val_normal)

                # Abnormal validation MSE
                recon_abnormal, _, _, _ = self.forward(self.x_val_abnormal)
                mse_abnormal = F.mse_loss(recon_abnormal, self.y_val_abnormal)

                ratio = mse_abnormal / (mse_normal + 1e-8)

            self.log('val_mse_normal', mse_normal, prog_bar=True, logger=True)
            self.log('val_mse_abnormal', mse_abnormal, prog_bar=True, logger=True)
            self.log('val_ratio', ratio, prog_bar=True, logger=True)
        finally:
            self.train()

    # Optimizers configuration
    # AdamW - weight decay decoupled from gradient updates -> better generalization (less overfit than Adam), Stability increased (eps = 1e-4)
    # CosineAnnealingWarmRestarts - control the learning rate (varies in speed waves with cosine decay over T_0 epochs until reaching eta_min, when T_0 finishes it restarts lr)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                 weight_decay=self.hparams.weight_decay, 
                                 eps=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1)
        return [opt], [sch]
    
    @staticmethod
    def _build_sample_weights(pids):
        """Compute per-sample weights so every pump is equally represented.

        Weight for each sample = 1 / (number of samples from that pump).
        The WeightedRandomSampler then draws proportionally, ensuring
        balanced representation without discarding any data.
        """
        if pids is None:
            return None
        pids_arr = np.asarray(pids)
        unique, counts = np.unique(pids_arr, return_counts=True)
        weight_map = {pid: 1.0 / cnt for pid, cnt in zip(unique, counts)}
        weights = np.array([weight_map[p] for p in pids_arr], dtype=np.float64)
        return torch.from_numpy(weights)

    # Dataloaders for training and test
    def train_dataloader(self):
        dataset = PumpDataset(self.x_train, self.y_train)
        on_cpu = not self.x_train.is_cuda

        # Pump-balanced sampling: draw from all pumps equally per epoch
        if self._sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True,
            )
            return DataLoader(
                dataset, batch_size=self.hparams.batch_size,
                pin_memory=on_cpu and should_pin_memory(),
                sampler=sampler,  # mutually exclusive with shuffle
            )

        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          pin_memory=on_cpu and should_pin_memory(), shuffle=True)