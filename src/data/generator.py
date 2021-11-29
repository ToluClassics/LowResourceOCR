"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

from itertools import groupby
from torch.utils.data import Dataset
import torch
import cv2
import random

import src.data.preprocess as pp

#import preprocess as pp
import os
import numpy as np
import unicodedata
import torchvision.transforms as T
from PIL import Image
from numpy import asarray


class DataGenerator(Dataset):
    """Generator class with data streaming"""

    def __init__(self, source, charset, transform, lang, max_len=200):
        self.transform = transform

        self.source = os.path.join(source, f"{lang}_image")
        self.images = os.listdir(self.source)
        self.images = [image for image in self.images if image.endswith(".jpg")]
        random.shuffle(self.images)

        self.image_dataset = [
            asarray(Image.open(os.path.join(self.source, img))) for img in self.images
        ]

        with open(os.path.join(source, f"{lang}_target.txt"), "r") as f:
            text = f.read()
        text = text.split("\n")
        text = [item.split() for item in text if len(item.strip()) > 1]
        self.gt = {k[0]: " ".join(k[1:]) for k in text}

        self.max_len = max_len #max([len(item.strip()) for item in list(self.gt.values())])
        self.tokenizer = Tokenizer(charset, self.max_len, lang=lang)

        self.size = len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        img = os.path.join(self.source, img)

        img = pp.preprocess(img, (1024, 128, 1))
        # making image compatible with resnet
        img = np.repeat(img[..., np.newaxis], 3, -1)
        img = pp.normalization(img)

        if self.transform is not None:
            img = self.transform(img)

        self.gt[self.images[i]] = pp.text_standardize(self.gt[self.images[i]])
        y_train = self.tokenizer.encode(self.gt[self.images[i]])

        # padding till max length
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        return img, gt

    def __len__(self):
        return self.size


class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length, lang):
        self.PAD_TK, self.UNK_TK, self.SOS, self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = (
            [self.PAD_TK] + [self.UNK_TK] + [self.SOS] + [self.EOS] + list(chars)
        )
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length
        self.lang = lang

    def encode(self, text):
        """Encode text to vector"""
        if self.lang == 'eng':
            text = (
                unicodedata.normalize("NFKD", text)
                .encode("ASCII", "ignore")
                .decode("ASCII")
            )
        else:
            text = (
                unicodedata.normalize("NFC", text)
            )

        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        text = ["SOS"] + list(text) + ["EOS"]
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


if __name__ == "__main__":
    charset = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~̣ṄṅẁỊịỌọỤụ "
    split = "train"
    transform = T.Compose([T.ToTensor()])
    dg = DataGenerator("raw_data/trdg", charset, transform, lang='igbo')
    train_loader = torch.utils.data.DataLoader(dg, batch_size=32, shuffle=False, num_workers=2)
    for i, (a, b) in enumerate(train_loader):
        print(i)
    '''orig_text = "iru igwe di ọkpala da akakpọ dịnyelụ anụ ugboko"
    tokenizer = Tokenizer(chars=charset, max_text_length= 183, lang="igbo")
    encode = tokenizer.encode(orig_text)
    text = tokenizer.decode(encode)
    print(len(text))
    print(len(encode))
    print(len(orig_text))
    print(text)'''