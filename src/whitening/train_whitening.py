from sklearn.metrics import accuracy_score

def evaluate(model, tokenizer, threshold=0.7):
    print("\nRunning evaluation on QQP validation set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    dataset = load_dataset("glue", "qqp")["validation"]
    texts1 = list(dataset["question1"])
    texts2 = list(dataset["question2"])
    labels = list(dataset["label"])

    enc1 = tokenizer(texts1, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc2 = tokenizer(texts2, truncation=True, padding=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        z1 = model(enc1["input_ids"].to(device), enc1["attention_mask"].to(device))
        z2 = model(enc2["input_ids"].to(device), enc2["attention_mask"].to(device))

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sims = (z1 * z2).sum(dim=1).cpu()

    preds = (sims > threshold).long().numpy()
    acc = accuracy_score(labels, preds)

    print(f"Evaluation Accuracy @ threshold {threshold}: {acc * 100:.2f}%")

if __name__ == "__main__":
    train()

    # Load model for evaluation
    tokenizer = AutoTokenizer.from_pretrained("whitening_roberta")
    model = WhitenedRoberta("whitening_roberta")

    evaluate(model, tokenizer, threshold=0.7)
