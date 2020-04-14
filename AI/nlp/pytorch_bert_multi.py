"""

# Links
https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
https://engineering.wootric.com/when-bert-meets-pytorch

"""
import copy
import json
import os
import numpy as np
import pandas as pd
from pypchutils.generic import create_logger
from transformers import BertTokenizer, BertConfig, BertModel
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = create_logger(__name__)

# bert parameters
MAX_LEN = 256  # max is 512 for BERT
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)


class TextDataset(data.Dataset):
    """
    a custom dataset class that uses the BERT tokenizer to map batches of text data to a tensor of its
    respective BERT model vocabulary IDs, while also adding the right amount of padding. This process is similar to
    constructing any custom dataset class in pytorch by inheriting the base Dataset class, and
    modifying the __getitem__ function.Below is the custom dataset class:
    """

    def __init__(self, xy: list, tokenizer, max_seq_length: int = MAX_LEN):
        self.xy = xy
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        """Return the tensors for the review and positive/negative labels
        """
        tokenized = self.tokenizer.tokenize(self.xy[0][index])
        tokenized = tokenized[: self.max_seq_length] if len(tokenized) > self.max_seq_length else tokenized

        ids = self.tokenizer.convert_tokens_to_ids(tokenized)

        padding = [0] * (self.max_seq_length - len(ids))
        ids += padding
        assert len(ids) == self.max_seq_length

        ids = torch.tensor(ids)

        labels = [torch.from_numpy(np.array(self.xy[1][index]))]
        return ids, labels[0]

    def __len__(self):
        return len(self.xy[0])


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels: int = 2, pre_trained: str = "bert-base-uncased"):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pre_trained)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        #
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids: list = None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def load_train_test_sets(filepath: str = "IMDB Dataset.csv", test_size: float = 0.10, random_state: int = 42):
    data = pd.read_csv(filepath)
    X, y = data["review"], data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def main(
    data_file: str,
    pre_trained: str = "bert-base-uncased",
    num_labels: int = 2,
    batch_size: int = 16,
    epochs: int = 5,
    best_loss: int = 100,
    lrlast: float = 0.001,
    lrmain: float = 0.00001,
):
    tokenizer = BertTokenizer.from_pretrained(pre_trained)
    # tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", pre_trained)

    X_train, X_test, y_train, y_test = load_train_test_sets(data_file)
    train_lists = [X_train, y_train]
    test_lists = [X_test, y_test]
    train_dataset = TextDataset(train_lists, tokenizer=tokenizer)
    test_dataset = TextDataset(test_lists, tokenizer=tokenizer)

    dataloaders_dict = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    }
    dataset_sizes = {"train": len(train_lists[0]), "val": len(test_lists[0])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is {device}")

    model = BertForSequenceClassification(config=config, num_labels=num_labels)
    best_model_wts = copy.deepcopy(model.state_dict())
    optim1 = optim.Adam(
        [{"params": model.bert.parameters(), "lr": lrmain}, {"params": model.classifier.parameters(), "lr": lrlast}]
    )
    # optim1 = optim.Adam(model.parameters(), lr=0.001)  #,momentum=.9)

    # Observe that all parameters are being optimized
    optimizer = optim1
    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch_index in range(epochs):
        logger.info("Epoch {}/{}".format(epoch_index, epochs - 1))
        logger.info("-" * 10)

        # Each epoch_index has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                exp_lr_scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss, sentiment_corrects = 0.0, 0
            for inputs, sentiment in dataloaders_dict[phase]:  # Iterate over data.
                inputs = inputs.to(device)
                sentiment = sentiment.to(device)
                optimizer.zero_grad()  # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            logger.info("{} total loss: {:.4f} ".format(phase, epoch_loss))
            logger.info("{} sentiment_acc: {:.4f}".format(phase, sentiment_acc))
            if phase == "val" and epoch_loss < best_loss:
                logger.info("saving with loss of {}".format(epoch_loss), "improved over previous {}".format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "bert_model_test.pth")
        logger.info("Printing out the model after epoch: %d\n%s" % (epoch_index, str(model)))

    logger.info("Best val Acc: {:4f}".format(float(best_loss)))
    model.load_state_dict(best_model_wts)  # Load best model weights. This is the final model
    logger.info("Printing out the final model:\n%s" % (str(model)))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default=f"{os.environ.get('HOME')}/Google Drive/xheng/data/imdb_reviews_dataset.csv"
    )
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
