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
    desired_percentage=0.1,
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
    src_vocab_size=13 + 1,
    tgt_vocab_size=6 + 1,  # 3 for <PAD>, <SOS>, <EOS> -> 1 for <PAD>
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
        loss = criterion(output.view(-1, 6 + 1), tgt.view(-1))
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
def get_seq_len(seq, pad_idx=13):
    for i, token in enumerate(seq[0]):
        if token == pad_idx:
            return i-1

pred_true_pairs_tgt = {}
pred_true_pairs_src = {}

with torch.no_grad():
    for i, (src, tgt) in enumerate(test_dataloader):
        src_len = get_seq_len(src,13)
        tgt_len = get_seq_len(tgt,6)

        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt)
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
    token_acc[key] , sequence_acc[key] = get_accuracy(pred_true_pairs_tgt[key][0][0],pred_true_pairs_tgt[key][0][1],train_pad_idxs[1],train_pad_idxs[1])

print(token_acc)

# plot the accuracy for each sequence length
from matplotlib import pyplot as plt
plt.bar(token_acc.keys(), token_acc.values())
plt.xlabel("Target Sequence Length")
plt.ylabel("Token Accuracy")
plt.title("Token accuracy")
plt.show()

