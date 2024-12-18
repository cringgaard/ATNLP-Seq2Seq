from scan_dataset import make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

splits = [
    "p1",
    "p2",
    "p4",
    "p8",
    "p16",
    "p32",
    "p64",
    "full",
]

token_acc = []
sequence_acc = []

EPOCHS = 1
device = get_device()
hyperparameters = experiment_hyperparameters["1"]


for split in splits:

    # Load the dataset
    dataloader_res = make_dataloader(
        f"experiment_1_{split}_train.txt",
        hyperparameters["BATCH_SIZE"],
        60,
        desired_percentage=1,
        upscale=False,
    )
    train_dataloader = dataloader_res["dataloader"]
    train_pad_idxs = dataloader_res["pad_idxs"]

    test_dataloader_res = make_dataloader(
        f"experiment_1_{split}_test.txt",
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

    # Evaluation loop
    model.eval()
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

            # Calculate token accuracy
            correct_tokens = (output == tgt).sum().item()
            total_tokens += output.numel()
            total_token_accuracy += correct_tokens

            # Calculate sequence accuracy
            correct_sequences = (output == tgt).all(dim=-1).sum().item()
            total_sequences += output.shape[0]
            total_sequence_accuracy += correct_sequences

    token_acc.append(total_token_accuracy / total_tokens)
    sequence_acc.append(total_sequence_accuracy / total_sequences)

from matplotlib import pyplot as plt
plt.bar(splits, token_acc)
plt.title("Token accuracy")
plt.show()
