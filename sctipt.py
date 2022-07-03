#! /usr/bin/env python

# !git clone https://github.com/ToluClassics/LowResourceOCR.git 


# !pip install -q transformers
# !pip install -q datasets jiwer
import os, argparse
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.notebook import tqdm

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW

from datasets import load_metric

"""
features we may want to pass as arguments include:
text_path, 
ImageFolder_Path, 
split_size, 
max_target_length, 
no_epochs,
train_batch_size
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the parser
the_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
the_parser.add_argument('--text_path',
                       metavar='path',
                       type=str,
                       help='the path to dataframe .txt file')
the_parser.add_argument('--dir_path',
                       metavar='dpath',
                       type=str,
                       help='the path to dataframe .txt file')
the_parser.add_argument('--split_size',
                       type=float,
                       help='train_test split size for validation',
                       default=0.2)
the_parser.add_argument('--max_target_length',
                       type=int,
                       help='train_test split size for validation',
                       default=256)
the_parser.add_argument('--no_epochs',
                       type=int,
                       help='no of epochs in training',
                       default=10)
the_parser.add_argument('--training_batch_size',
                       type=int,
                       help='training batch size',
                       default=4)

# Execute parse_args()
args = the_parser.parse_args()

txt_path = args.text_path
dir_path = args.dir_path
split_size = args.split_size
max_target_length = args.max_target_length
no_epochs = args.no_epochs
train_batch_size = args.training_batch_size

if not os.path.isdir(txt_path):
    print('The file specified does not exist')
    sys.exit()


df = pd.read_fwf(txt_path, header=None)

df['filename']= df[0].str.split(' ').str[0]
df[2] = df['filename']
for i in range(len(df)):

  alltext = df.iloc[i, 0].split(' ')[1:]
  df.iloc[i, 2] = " ".join(alltext)

df.drop(columns=0, inplace=True)

df.rename(columns={0: "filename", 2: "text"}, inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=256):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['filename'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir=dir_path,
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=dir_path,
                           df=test_df,
                           processor=processor)
encoding = train_dataset[0]

image = Image.open(train_dataset.root_dir + train_df['filename'][0]).convert("RGB")

labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=train_batch_size)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = load_metric("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_dataloader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(eval_dataloader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained(".")

