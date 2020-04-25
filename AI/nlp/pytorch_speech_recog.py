"""
# Link
https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch

"""
import json
import os
import pandas as pd
from pypchutils.generic import create_logger
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

logger = create_logger(__name__)


def download_data(folder: str = "{os.environ.get('HOME')}/Google Drive/xheng/data", download: bool = False):
    train_dataset = torchaudio.datasets.LIBRISPEECH(folder, url="train-clean-100", download=download)
    test_dataset = torchaudio.datasets.LIBRISPEECH(folder, url="test-clean", download=download)
    return train_dataset, test_dataset


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
         1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = " "

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map[""]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("", " ")


def main(
    data_folder: str = "{os.environ.get('HOME')}/Google Drive/xheng/data",
    download: bool = False,
    batch_size: int = 100,
    epochs: int = 20,
    learning_rate=0.01,
):
    train_dataset, test_dataset = download_data(folder=data_folder, download=download)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default=f"{os.environ.get('HOME')}/Google Drive/xheng/data",
        help=f"{os.environ.get('HOME')}/Google Drive/xheng/data,{os.environ.get('HOME')}/data",
    )
    parser.add_argument("--download", default="false")

    # Parse the cmd line args
    args = vars(parser.parse_args())
    args["download"] = args["download"].lower() == "true"
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
