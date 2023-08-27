"""
https://visualstudiomagazine.com/articles/2021/11/16/fine-tune-nlp-model.aspx

# fine-tune HF pretrained model for IMDB

# zipped raw data at: https://ai.stanford.edu/~amaas/data/sentiment/

"""
from datetime import datetime
import numpy as np
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from transformers import AdamW, DistilBertForSequenceClassification
from transformers import logging

logging.set_verbosity_error()  # suppress wordy warnings

device = torch.device("cpu")


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, reviews_lst, labels_lst):
        self.reviews_lst = reviews_lst  # list of token IDs
        self.labels_lst = labels_lst  # list of 0-1 ints

    def __getitem__(self, idx):
        item = {}  # [input_ids] [attention_mask] [labels]
        for key, val in self.reviews_lst.items():
            item[key] = torch.tensor(val[idx]).to(device)
        item["labels"] = torch.tensor(self.labels_lst[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels_lst)


def read_imdb(root_dir):
    reviews_lst = []
    labels_lst = []
    root_dir = Path(root_dir)
    for label_dir in ["pos", "neg"]:
        for f_handle in (root_dir / label_dir).iterdir():
            reviews_lst.append(f_handle.read_text(encoding="utf-8"))
            if label_dir == "pos":
                labels_lst.append(1)
            else:
                labels_lst.append(0)
    return (reviews_lst, labels_lst)  # lists of strings


def main(proj_root_dir, epochs: int = 3):
    print("Begin fine-tune for IMDB sentiment")
    torch.manual_seed(1)
    np.random.seed(1)

    # Load raw IMDB train data into memory
    print("\nLoading IMDB train data subset into memory...")
    train_reviews, train_labels = read_imdb(f"{proj_root_dir}/train")

    # consider creating validation set here
    #
    #

    # Tokenize the raw data reviews text
    print("\nTokenizing training text...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_tokens = tokenizer(train_reviews, truncation=True, padding=True)  # token IDs and mask

    # Load tokenized text and labels into PyTorch Dataset
    print("\nLoading tokenized text into Pytorch Dataset")
    train_dataset = IMDbDataset(train_tokens, train_labels)

    # Load (possibly cached) pretrained HF model
    print("\nLoading pre-trained DistilBERT model ")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.to(device)
    model.train()  # set at training mode

    # Fine-tune / train model using standard PyTorch
    print("Loading Dataset with batch_size: 10 ")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    print(f"\nFine-tuning the model. It's now {datetime.now()}")
    optim = AdamW(model.parameters(), lr=5.0e-5)  # weight decay
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optim.zero_grad()

            input_ids = batch["input_ids"]  # tensor
            attn_mask = batch["attention_mask"]  # tensor
            labels = batch["labels"]  # tensor

            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.item()  # accumulate batch loss
            loss.backward()
            optim.step()
            if batch_idx % 20 == 0:
                print(
                    "batch_idx: %5d, curr batch loss: %0.4f. It is now: %s" % (batch_idx, loss.item(), datetime.now())
                )
        print("End of epoch no. %4d, epoch loss = %0.4f. Now is %s" % (epoch, epoch_loss, datetime.now()))
    print("Training is complete")

    # 6. save trained model weights and biases
    print("\nSaving tuned model state")
    model.eval()
    torch.save(model.state_dict(), f"{proj_root_dir}/models/imdb_state.pt")  # just state

    print("\nEnd pf demo")


if __name__ == "__main__":
    proj_root_dir = f"{os.getenv('HOME')}/dev/github/xinh3ng/data/aclImdb"
    main(proj_root_dir)
