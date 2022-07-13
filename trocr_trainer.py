#! /usr/bin/env python

import random
import wandb
import os, argparse
import sys
import copy
import pandas as pd
from datasets import load_metric
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm


wandb.init(project="LowResourceOCR", entity="toluclassics")
cer_metric = load_metric("cer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "raw_data/trdg/eng_image"

class OcrDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length):
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Takes configuration for Transformer OCR training"
    )
    parser.add_argument('--text_path',
                       metavar='path',
                       type=str,
                       help='the path to dataframe .txt file')
    parser.add_argument('--dir_path',
                          metavar='dpath',
                          type=str,
                          help='the path to data images')
    parser.add_argument('--model_name',
                          metavar='model_name',
                          type=str,
                          default="microsoft/trocr-base-printed")
    parser.add_argument('--processor_tokenizer',
                          metavar='processor_tokenizer',
                          type=str,
                          default=None)
    parser.add_argument('--split_size',
                          type=float,
                          help='train_test split size for validation',
                          default=0.2)
    parser.add_argument('--max_target_length',
                          type=int,
                          help='train_test split size for validation',
                          default=256)
    parser.add_argument('--no_epochs',
                          type=int,
                          help='no of epochs in training',
                          default=10)
    parser.add_argument('--training_batch_size',
                          type=int,
                          help='training batch size',
                          default=4)
    parser.add_argument('--learning_rate',
                          type=float,
                          help='Set Optimizer learning rate, default is',
                          default=5e-5)
    parser.add_argument('--max_length',
                          type=int,
                          help='max length',
                          default=64)                    
    parser.add_argument('--early_stopping',
                          type=bool,
                          help='To activate early stopping or not',
                          default=True)
    parser.add_argument('--no_repeat_ngram_size',
                          type=int,
                          help='Number of repeat ngram size',
                          default=3)
    parser.add_argument('--length_penalty',
                          type=float,
                          help='model config length penalty',
                          default=2.0)    
    parser.add_argument('--num_beams',
                          type=int,
                          help='model config number of beams',
                          default=4)
    parser.add_argument('--model_outputdir',
                       type=str,
                       help='path to save model to',
                       default='.')  
    parser.add_argument('--seed',
                       type=int,
                       help='random seed',
                       default=42)                             
    return parser

def compute_cer(pred_ids, label_ids, processor):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def gen_training_dataframe(file_path: str):
    df = pd.read_fwf(file_path, header=None)

    df['filename']= df[0].str.split(' ').str[0]
    df[2] = df['filename']
    for i in range(len(df)):
        alltext = df.iloc[i, 0].split(' ')[1:]
        df.iloc[i, 2] = " ".join(alltext)

    df.drop(columns=0, inplace=True)

    df.rename(columns={0: "filename", 2: "text"}, inplace=True)

    return df

def process_image(image, processor, model):
    # prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    # generate (no beam search)
    generated_ids = model.generate(pixel_values)
    # decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def main():

    # Create the parser
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
                        
    assert (os.path.isfile(args.text_path) == True), 'The file specified does not exist'

    df = gen_training_dataframe(args.text_path)

    train_df, test_df = train_test_split(df, test_size=args.split_size)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    train_dataset = OcrDataset(root_dir=args.dir_path,
                            df=train_df,
                            processor=processor,
                            max_target_length=args.max_target_length)
    eval_dataset = OcrDataset(root_dir=args.dir_path,
                            df=test_df,
                            processor=processor,
                            max_target_length=args.max_target_length)
    encoding = train_dataset[0]

    wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.no_epochs,
            "batch_size": args.training_batch_size,
            "processor_tokenizer": args.processor_tokenizer,
            "num_beams": args.num_beams,
            "text_path": args.text_path,
            "model_name": args.model_name,
            "early_stopping": args.early_stopping,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
            }

    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.training_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.training_batch_size)

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = args.early_stopping
    model.config.no_repeat_ngram_size = args.no_repeat_ngram_size
    model.config.length_penalty = args.length_penalty
    model.config.num_beams = args.num_beams

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    text_table = wandb.Table(columns=["epoch", "original_label", "generated_label"])

    for epoch in range(args.no_epochs):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k, v in batch.items():
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
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"], processor=processor)
                valid_cer += cer 

        print("Validation CER:", valid_cer / len(eval_dataloader))
        
        wandb.log({"loss": loss, "epoch": epoch, "validation_character_error_rate": valid_cer / len(eval_dataloader) })


        #predict during training
        with torch.no_grad():
            sampled_files = os.listdir(data_path)
            random.shuffle(sampled_files)
            for i, file in enumerate(sampled_files):
                test_image = Image.open(os.path.join(data_path, file)).convert("RGB")
                generated_text = process_image(test_image, processor, model)
                print(f"[Predict While Training-> Generated Text] :: {generated_text}")
                print(f"[Predict While Training-> Groundtruth Text] :: {' '.join(file.split('_')[:-1]).upper()}")
                print("==============================================================================================\n")

                text_table.add_data(epoch, ' '.join(file.split('_')[:-1]).lower(), generated_text.lower())

                if i > 5:
                    break
            wandb.log({"generated_text" : text_table})


    model.save_pretrained(args.model_outputdir)
    wandb.log_artifact(args.model_outputdir, name=args.model_outputdir, type=args.model_name) 



if __name__ == "__main__":
    main()