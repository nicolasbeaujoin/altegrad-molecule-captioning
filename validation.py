from torchmetrics.text import BLEUScore, BERTScore

# Setup Metrics
# BLEU-4 matches n-grams up to order 4 [cite: 85]
bleu_metric = BLEUScore(n_gram=4)
# BERTScore uses a pre-trained model (challenge uses RoBERTa-base [cite: 86])
bert_metric = BERTScore(model_name_or_path="roberta-base")

# Load Validation Data
val_dataset = PreprocessedGraphDataset("validation_graphs.pkl")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

preds = []
targets = []

model.eval()
with torch.no_grad():
    for data in tqdm(val_loader):
        data = data.to(device)

        # Generate
        pred_text = model.generate_caption(data)

        # Get Ground Truth
        # data.description contains the real text [cite: 40]
        true_text = data.description[0]

        preds.append(pred_text)
        targets.append(true_text)

# Calculate Scores
# BLEU expects a list of predictions and a list of list of references
bleu_score = bleu_metric(preds, [[t] for t in targets])
print(f"Validation BLEU-4: {bleu_score.item()}")

# BERTScore returns precision, recall, and F1
bert_scores = bert_metric(preds, targets)
print(f"Validation BERTScore F1: {bert_scores['f1'].mean().item()}")
