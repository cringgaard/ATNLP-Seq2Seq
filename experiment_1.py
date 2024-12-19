from scan_dataset import make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json

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

device = get_device()
hyperparameters = experiment_hyperparameters["1"]
results = {}
EPOCHS = 5


for split in splits:
    train_loader_dict = make_dataloader(
        f"experiment_1_{split}_train.txt",
        hyperparameters["BATCH_SIZE"],
        50,
        1,
        True,
        100000,
    )
    test_loader_dict = make_dataloader(
        f"experiment_1_{split}_test.txt",
        hyperparameters["BATCH_SIZE"],
        50,
        1,
        False,
        100000,
    )

    src_pad_idx = train_loader_dict["pad_idxs"][0]
    tgt_pad_idx = train_loader_dict["pad_idxs"][1]
    src_sos_idx = src_pad_idx + 1
    src_eos_idx = src_pad_idx + 2
    tgt_sos_idx = tgt_pad_idx + 1
    tgt_eos_idx = tgt_pad_idx + 2

    model = Transformer(
        src_vocab_size=13 + 3,
        tgt_vocab_size=6 + 3,  # 3 for <PAD>, <SOS>, <EOS>
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        emb_dim=hyperparameters["EMB_DIM"],
        num_layers=hyperparameters["N_LAYERS"],
        num_heads=hyperparameters["N_HEADS"],
        forward_dim=hyperparameters["FORWARD_DIM"],
        dropout=hyperparameters["DROPOUT"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["LEARNING_RATE"])

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for i, batch in enumerate(train_loader_dict["dataloader"]):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1)
            )
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=hyperparameters["GRAD_CLIP"]
            )


            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.eval()
    total_token_acc = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader_dict["dataloader"]):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            output = model(src, tgt[:, :-1])
            pred_tokens = output.argmax(2)
            # Pred with <SOS> token
            pred_tokens = torch.cat([tgt[:, :1], pred_tokens], dim=-1)
            token_acc = (pred_tokens == tgt).sum()
            total_token_acc += token_acc.item()
            total_tokens += tgt.numel()

    results[split] = total_token_acc / total_tokens
    print(f"Split: {split}, Token Accuracy: {results[split]}")

if not os.path.exists("results"):
    os.makedirs("results")


with open("results/experiment_1.json", "w") as f:
    json.dump(results, f)

print(results)
