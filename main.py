from scan_dataset import make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Epochs is not listed in the hyperparameters, so I'll use 15 as a default
EPOCHS = 1

device = get_device()
# device = torch.device("cpu")

# I'll use the first experiment hyperparameters for this task
hyperparameters = experiment_hyperparameters["1"]

# Load the dataset
dataloader_res = make_dataloader(
    "tasks_train_simple_p1.txt",
    hyperparameters["BATCH_SIZE"],
    80,
    desired_percentage=0.01,
    upscale=False,
)
train_dataloader = dataloader_res["dataloader"]
train_pad_idxs = dataloader_res["pad_idxs"]

test_dataloader_res = make_dataloader(
    "tasks_test_simple_p1.txt",
    hyperparameters["BATCH_SIZE"],
    80,
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
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")


# Evaluation loop
model.eval()
# We are interested in per batch accuracy, as well as overall accuracy both for tokens and sequences
total_token_accuracy = 0
total_sequence_accuracy = 0
total_tokens = 0
total_sequences = 0

with torch.no_grad():
    for i, (src, tgt) in enumerate(test_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt)
        output = output.argmax(dim=-1)

        if i == 0:
            print("Output shape:", output.shape)
            print("Input example:", src[0])
            print("Output example:", output[0])
            print("Target example:", tgt[0])

        # Calculate token accuracy
        correct_tokens = (output == tgt).sum().item()
        total_tokens += output.numel()
        total_token_accuracy += correct_tokens

        # Calculate sequence accuracy
        correct_sequences = (output == tgt).all(dim=-1).sum().item()
        total_sequences += output.shape[0]
        total_sequence_accuracy += correct_sequences

        print(
            f"Batch {i}, Token Accuracy: {correct_tokens / output.numel():.6f}, Sequence Accuracy: {correct_sequences / output.shape[0]:.6f}"
        )

print(
    f"Overall Token Accuracy: {total_token_accuracy / total_tokens:.6f}, Overall Sequence Accuracy: {total_sequence_accuracy / total_sequences:.6f}"
)


# Save the model
if not os.path.exists("models"):
    os.makedirs("models")

torch.save(model.state_dict(), "models/experiment1.pt")
