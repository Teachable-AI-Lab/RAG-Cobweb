import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class DocumentEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', proj_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = ProjectionHead(input_dim=self.encoder.config.hidden_size, proj_dim=proj_dim)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = output.last_hidden_state[:, 0]  # [CLS]
        return self.proj(cls_embed)


def cu_similarity(z1, z2, eps=1e-8):
    var1 = z1.var(dim=0, unbiased=False) + eps
    var2 = z2.var(dim=0, unbiased=False) + eps
    var_comb = ((var1 + var2) / 2)

    gain = 0.5 * (torch.log(var_comb) - 0.5 * (torch.log(var1) + torch.log(var2)))
    return gain.sum()


def cu_contrastive_loss(z1, z2, temperature=0.05):
    B = z1.size(0)
    logits = torch.zeros(B, B).to(z1.device)

    for i in range(B):
        for j in range(B):
            logits[i, j] = cu_similarity(z1[i:i+1], z2[j:j+1])

    logits /= temperature
    labels = torch.arange(B).to(z1.device)
    return F.cross_entropy(logits, labels)


def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4):
    # Invariance
    sim_loss = F.mse_loss(z1, z2)

    # Variance
    def variance_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1 - std))

    var_loss = variance_term(z1) + variance_term(z2)

    # Covariance
    def covariance_term(z):
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / z.shape[1]

    cov_loss = covariance_term(z1) + covariance_term(z2)

    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss


def load_qqp_pairs(tokenizer, max_length=64):
    dataset = load_dataset("glue", "qqp")["train"]
    dataset = dataset.filter(lambda x: x["is_duplicate"] == 1)
    texts1 = dataset["question1"]
    texts2 = dataset["question2"]

    encodings1 = tokenizer(texts1, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encodings2 = tokenizer(texts2, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    return list(zip(encodings1["input_ids"], encodings1["attention_mask"],
                    encodings2["input_ids"], encodings2["attention_mask"]))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = DocumentEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    data = load_qqp_pairs(tokenizer)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            q1_ids, q1_mask, q2_ids, q2_mask = [x.to(device) for x in batch]
            z1 = model(q1_ids, q1_mask)
            z2 = model(q2_ids, q2_mask)

            loss_cu = cu_contrastive_loss(z1, z2)
            loss_vic = vicreg_loss(z1, z2)
            loss = loss_cu + loss_vic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
