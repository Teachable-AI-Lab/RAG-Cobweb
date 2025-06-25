import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np

class BetaVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256, beta: float = 4.0, dropout: float = 0.3, latent_dropout: float = 0.0):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_dropout = latent_dropout

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

        # Removed skip connection
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # removed for collapse prevention

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Optional: add small noise to encourage exploration
        z += 1e-3 * torch.randn_like(z)

        # Optional: apply latent dropout
        if self.latent_dropout > 0:
            z = F.dropout(z, p=self.latent_dropout, training=self.training)

        return z

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        h = self.decoder_dropout(h)
        recon = self.decoder_fc2(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, free_bits=0.5):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Apply free bits threshold
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)

        kl_div = kl_per_dim.sum(dim=1).mean()

        total_loss = recon_loss + self.beta * kl_div
        return total_loss, recon_loss, kl_div

    def save_model(self, path: str):
        torch.save({
            'model_state': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta
        }, path)

    @classmethod
    def load_model(cls, path: str):
        checkpoint = torch.load(path)
        model = cls(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            beta=checkpoint['beta']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model
    


def linear_beta_schedule(epoch, total_epochs, final_beta=1.0):
    return final_beta * min(1.0, epoch / total_epochs)

def cyclical_beta_schedule(epoch, cycle_length, final_beta=1.0):
    cycle_pos = epoch % cycle_length
    return final_beta * min(1.0, cycle_pos / cycle_length)

def inspect_latent_usage(model, data_tensor, batch_size=256):
    model.eval()
    z_vals = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            mu, _ = model.encode(batch)
            z_vals.append(mu.cpu().numpy())
    z_vals = np.concatenate(z_vals, axis=0)
    variances = np.var(z_vals, axis=0)
    print("\nLatent dimension variances:")
    relevant_dims = False
    for i, v in enumerate(variances):
        if v >= 0.01:
            relevant_dims = True
            print(f"Relevant Dim {i:02d}: {v:.6f}")

    if not relevant_dims:
        print("No relevant dimensions (all were below 0.01)")

def train_vae(
    model,
    data,
    epochs=20,
    lr=1e-3,
    beta_schedule='constant',
    warmup_epochs=5,
    cycle_length=10,
    final_beta=10.0,
    free_bits=0.5,
    device=None,
    kl_warn_threshold=0.01,
    kl_stall_patience=10
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    kl_history = []

    for epoch in trange(epochs, desc="Training", unit="epoch"):
        model.train()

        # Determine beta for this epoch
        if beta_schedule == 'linear':
            model.beta = final_beta * min(1.0, epoch / warmup_epochs)
        elif beta_schedule == 'cyclical':
            cycle_pos = epoch % cycle_length
            model.beta = final_beta * min(1.0, cycle_pos / cycle_length)
        else:
            model.beta = final_beta

        optimizer.zero_grad()
        recon, mu, logvar = model(data_tensor)

        loss, recon_loss, kl_loss = model.loss_function(
            recon, data_tensor, mu, logvar, free_bits=free_bits
        )

        loss.backward()
        optimizer.step()

        kl_history.append(kl_loss.item())

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
            f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | Beta: {model.beta:.4f}"
        )

        # Warn if KL is flat/zero for too long
        if len(kl_history) >= kl_stall_patience:
            recent_kl = kl_history[-kl_stall_patience:]
            if all(k < kl_warn_threshold for k in recent_kl):
                print(f"\n⚠️  Warning: KL divergence has stayed below {kl_warn_threshold} "
                      f"for {kl_stall_patience} epochs. Posterior collapse is likely.\n"
                      f"Try: lower free_bits, use smaller latent dim, remove skip connection, or warm up beta more slowly.\n")

    inspect_latent_usage(model, data_tensor)


if __name__ == "__main__":
    msmarco_passage_train_embs = np.load('data/msmarco_passage_train_embs.npy')
    msmarco_query_train_embs = np.load('data/msmarco_query_train_embs.npy')
    EMB_DIM = 384
    bvae_model = BetaVAE(
        input_dim=1024,
        latent_dim=EMB_DIM,         # very compressive
        hidden_dim=512,
        beta=0.0,              # will warm up
        dropout=0.1,
        latent_dropout=0.0     # you can try 0.1 if needed
    )

    train_vae(
        model=bvae_model,
        data=np.concatenate((msmarco_passage_train_embs, msmarco_query_train_embs), axis=0),
        epochs=10,
        lr=1e-3,
        beta_schedule='linear',
        warmup_epochs=30,
        cycle_length=5,
        final_beta=1,
        free_bits=0.0,              # force KL contribution
        kl_warn_threshold=0.01,
        kl_stall_patience=10,
        device=None
    )
