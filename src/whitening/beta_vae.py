import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange

class BetaVAETrainer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        beta: float = 4.0,
        dropout: float = 0.3,
        device: str = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

        # Learnable skip connection
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.to(self.device)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_orig=None):
        h = F.relu(self.decoder_fc1(z))
        h = self.decoder_dropout(h)
        recon = self.decoder_fc2(h)
        if x_orig is not None:
            recon = self.alpha * recon + (1 - self.alpha) * x_orig
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, free_bits=0.5):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
        kl_div = kl_per_dim.sum(dim=1).mean()
        total_loss = recon_loss + self.beta * kl_div
        return total_loss, recon_loss, kl_div

    def beta_schedule(self, epoch, strategy, total_epochs, final_beta, warmup_epochs=5, cycle_length=10):
        if strategy == 'linear':
            return final_beta * min(1.0, epoch / warmup_epochs)
        elif strategy == 'cyclical':
            cycle_pos = epoch % cycle_length
            return final_beta * min(1.0, cycle_pos / cycle_length)
        return final_beta  # constant

    def train_model(
        self,
        data: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        beta_schedule: str = 'constant',
        warmup_epochs: int = 5,
        cycle_length: int = 10,
        final_beta: float = 10.0,
        free_bits: float = 0.5,
        batch_size: int = 256
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

        for epoch in trange(epochs, desc="Training", unit="epoch"):
            self.train()
            self.beta = self.beta_schedule(epoch, beta_schedule, epochs, final_beta, warmup_epochs, cycle_length)

            optimizer.zero_grad()
            recon, mu, logvar = self(data_tensor)

            print(f"[Epoch {epoch+1}] mu mean: {mu.mean():.4f}, std: {mu.std():.4f} | logvar mean: {logvar.mean():.4f}, std: {logvar.std():.4f}")

            loss, recon_loss, kl_loss = self.loss_function(recon, data_tensor, mu, logvar, free_bits)
            loss.backward()
            optimizer.step()

            print(
                f"Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                f"KL: {kl_loss.item():.4f} | Beta: {self.beta:.4f}"
            )

        self.inspect_latent_usage(data_tensor, batch_size)

    def inspect_latent_usage(self, data_tensor, batch_size=256):
        self.eval()
        z_vals = []
        with torch.no_grad():
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i+batch_size]
                mu, _ = self.encode(batch)
                z_vals.append(mu.cpu().numpy())

        z_vals = np.concatenate(z_vals, axis=0)
        variances = np.var(z_vals, axis=0)

        print("\nLatent dimension variances:")
        found = False
        for i, v in enumerate(variances):
            if v >= 0.01:
                found = True
                print(f"  Relevant Dim {i:02d}: Var = {v:.6f}")
        if not found:
            print("  No relevant latent dimensions (all below 0.01 variance).")

    def save(self, path: str):
        torch.save({
            'model_state': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta,
            'dropout': self.dropout
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        checkpoint = torch.load(path, map_location=device or 'cpu')
        model = cls(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            beta=checkpoint['beta'],
            dropout=checkpoint['dropout'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model
