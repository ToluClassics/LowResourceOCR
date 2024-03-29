from trainer import train, evaluate
import time
import yaml
import torch
import os
import argparse
from tqdm import tqdm
from trainer import LabelSmoothing
from src.data.generator import Tokenizer, DataGenerator
import numpy as np
import torch.nn as nn
from src.model.model import make_model
from torch.utils.data import random_split
import torchvision.transforms as T

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda:0" if torch.cuda.is_available() == True else "cpu"

parser = argparse.ArgumentParser(description="Train a language OCR")
parser.add_argument("--lang", help="language to train in ", required=True)

args = parser.parse_args()

print(f"[INFO] Language to train OCR for is {args.lang}")

if args.lang == "eng":
    charset = config["eng_charset"]
elif args.lang == "yor":
    charset = config["yor_charset"]
elif args.lang == "igbo":
    charset = config["igbo_charset"]

print(f"[INFO] Model Parameters are : {config}")

batch_size = config["batch_size"]
num_epochs = config["num_epochs"]

tokenizer = Tokenizer(
    chars=config[f"{args.lang}_charset"],
    max_text_length=config[f"{args.lang}_max_text_len"],
    lang=args.lang,
)
transform = T.Compose([T.ToTensor()])

config["target_path"] = config["target_path"].replace("lang", args.lang)

print(f"[INFO] Model Parameters are : {config}")

print("[INFO] Generating Dataset Loader")
dataset = DataGenerator(
    source=config["source"],
    charset=config[f"{args.lang}_charset"],
    transform=transform,
    lang=args.lang,
)
train_test_split = [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
train_dataset, val_dataset = random_split(dataset, train_test_split)

print(f"[INFO] Length of training dataset is {len(train_dataset)}")
print(f"[INFO] Length of validation dataset is {len(val_dataset)}")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
)

print(f"[INFO] Load pretrained model...")

model = make_model(vocab_len=108)
model.load_state_dict(
    torch.load("run/checkpoint_weights_eng_trdg.pt", map_location=device)
)

if args.lang in ["yor", "igbo"]:
    model.vocab = nn.Linear(512, tokenizer.vocab_size)
    model.decoder = nn.Embedding(tokenizer.vocab_size, 512)
    model.to(device)

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

    for epoch in range(config["num_epochs"]):
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
            torch.save(model.state_dict(), config["target_path"])
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
