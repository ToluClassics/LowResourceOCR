from trainer import train, evaluate
import time
import yaml
import torch
import os
from tqdm import tqdm
from trainer import LabelSmoothing
from src.data.generator import Tokenizer, DataGenerator
import numpy as np
from src.model.model import make_model
from torch.utils.data import random_split

import torchvision.transforms as T

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config["batch_size"]
num_epochs = config["num_epochs"]


device = "cuda:0" if torch.cuda.is_available() == True else "cpu"
charset = config["charset"]

tokenizer = Tokenizer(charset)
transform = T.Compose([T.ToTensor()])

target_path = config["target_path"]

dataset = DataGenerator(source=config["source"], charset=charset, transform=transform)
train_dataset, val_dataset = random_split(dataset, config["train_test_split"])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)


model = make_model(vocab_len=tokenizer.vocab_size)
model.to(device)

model.load_state_dict(torch.load("run/checkpoint_weights_trdg.pt"))

# train model
criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=0.1)
criterion.to(device)
lr = config["lr"]  # learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    best_valid_loss = np.inf
    c = 0
    print("[INFO] Training Begin ......")
    for epoch in range(num_epochs):
        print(
            f"Epoch: {epoch + 1:02}", "learning rate{}".format(scheduler.get_last_lr())
        )

        start_time = time.time()
        train_loss = train(
            model, criterion, optimizer, scheduler, train_loader, tokenizer
        )
        valid_loss = evaluate(model, criterion, val_loader, tokenizer)

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        c += 1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), target_path)
            c = 0

        if c > 4:
            # decrease lr if loss does not decreases after 5 steps
            scheduler.step()
            c = 0

        print(f"Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Val   Loss: {valid_loss:.3f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
