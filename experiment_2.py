from scan_dataset import make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

EPOCHS = 1
device = get_device()
hyperparameters = experiment_hyperparameters["2"]

# Load the dataset
dataloader_res = make_dataloader(
    f"experiment_2_full_train.txt",
    hyperparameters["BATCH_SIZE"],
    60,
    desired_percentage=1,
    upscale=False,
)
train_dataloader = dataloader_res["dataloader"]
train_pad_idxs = dataloader_res["pad_idxs"]

test_dataloader_res = make_dataloader(
    f"experiment_2_full_test.txt",
    hyperparameters["BATCH_SIZE"],
    60,
    desired_percentage=1,
    upscale=False,
)
test_dataloader = test_dataloader_res["dataloader"]

# Initialize the model
model = Transformer(
    src_vocab_size=13 + 3,
    tgt_vocab_size=6 + 3,  # 3 for <PAD>, <SOS>, <EOS>
    src_pad_idx=train_pad_idxs[0],
    tgt_pad_idx=train_pad_idxs[1],
    emb_dim=hyperparameters["EMB_DIM"],
    num_layers=hyperparameters["N_LAYERS"],
    num_heads=hyperparameters["N_HEADS"],
    forward_dim=hyperparameters["FORWARD_DIM"],
    dropout=hyperparameters["DROPOUT"],
).to(device)

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["LEARNING_RATE"])

# Initialize the loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, 6 + 3), tgt.view(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=hyperparameters["GRAD_CLIP"]
        )

        optimizer.step()

        total_loss += loss.item()

def get_accuracy(output, tgt, output_pad_idx, tgt_pad_idx):
    # Calculate token accuracy in places where tgt and output are not padding tokens
    correct_tokens = ((output != output_pad_idx) & (output == tgt)).sum().item()
    total_tokens = (tgt != tgt_pad_idx).sum().item()
    total_token_accuracy = correct_tokens / total_tokens

    # Calculate sequence accuracy
    correct_sequences = (output == tgt).all(dim=-1).sum().item()
    total_sequences = output.shape[0]
    total_sequence_accuracy = correct_sequences / total_sequences

    return total_token_accuracy, total_sequence_accuracy

# Evaluation loop
model.eval()

# to count length of target and source sequences without padding
def get_seq_len(seq, pad_idx):
    for i, token in enumerate(seq[0]):
        if token == pad_idx:
            return i

pred_true_pairs_tgt = {}
pred_true_pairs_src = {}

with torch.no_grad():
    for i, (src, tgt) in enumerate(test_dataloader):
        src_len = get_seq_len(src,train_pad_idxs[0])
        tgt_len = get_seq_len(tgt,train_pad_idxs[1])

        src = src.to(device)
        tgt = tgt.to(device)
        SOS_token = torch.tensor([7]).to(device)
        # Generate from <SOS> token
        batched_SOS = SOS_token.repeat(tgt.shape[0], 1)
        output = model(src, batched_SOS)
        output = output.argmax(dim=-1)

        if pred_true_pairs_tgt.get(tgt_len) is None:
            pred_true_pairs_tgt[tgt_len] = []
        if pred_true_pairs_src.get(src_len) is None:
            pred_true_pairs_src[src_len] = []
        pred_true_pairs_tgt[tgt_len].append((output,tgt))
        pred_true_pairs_src[src_len].append((output,tgt))
    

token_acc = {}
sequence_acc = {}
for key in pred_true_pairs_tgt.keys():
    preds = []
    tgts = []
    for pred, tgt in pred_true_pairs_tgt[key]:
        preds.append(pred)
        tgts.append(tgt)
    preds = torch.cat(preds)
    tgts = torch.cat(tgts)
    if key > 45:
        print("pred",preds)
        print("true",tgts)
    token_acc[key], sequence_acc[key] = get_accuracy(preds, tgts, train_pad_idxs[1], train_pad_idxs[1])

# plot the accuracy for each sequence length
from matplotlib import pyplot as plt
plt.bar(token_acc.keys(), token_acc.values())
plt.xlabel("Target Sequence Length")
plt.ylabel("Token Accuracy")
plt.title("Token accuracy")
plt.show()

plt.bar(sequence_acc.keys(), sequence_acc.values())
plt.xlabel("Target Sequence Length")
plt.ylabel("Sequence Accuracy")
plt.title("Sequence accuracy")
plt.show()

token_acc = {}
sequence_acc = {}
for key in pred_true_pairs_src.keys():
    preds = []
    tgts = []
    for pred, tgt in pred_true_pairs_src[key]:
        preds.append(pred)
        tgts.append(tgt)
    preds = torch.cat(preds)
    tgts = torch.cat(tgts)
    if key == 48:
        print("pred",preds)
        print("true",tgts)
    token_acc[key], sequence_acc[key] = get_accuracy(preds, tgts, train_pad_idxs[1], train_pad_idxs[1])

# plot the accuracy for each sequence length
from matplotlib import pyplot as plt
plt.bar(token_acc.keys(), token_acc.values())
plt.xlabel("Target Sequence Length")
plt.ylabel("Token Accuracy")
plt.title("Token accuracy")
plt.show()

plt.bar(sequence_acc.keys(), sequence_acc.values())
plt.xlabel("Target Sequence Length")
plt.ylabel("Sequence Accuracy")
plt.title("Sequence accuracy")
plt.show()



