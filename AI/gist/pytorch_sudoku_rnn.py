"""

"""
import os
import pandas as pd
from pypchutils.generic import create_logger
import torch
import torch.utils.data as data
import torch.nn as nn

logger = create_logger(__name__)


def create_sudoku_tensors(df: pd.DataFrame, train_split: float = 0.5):

    def one_hot_encode(s):
        zeros = torch.zeros((1, 81, 9), dtype=torch.float)
        for a in range(81):
            zeros[0, a, int(s[a]) - 1] = 1 if int(s[a]) > 0 else 0
        return zeros

    quizzes_t = df.quizzes.apply(one_hot_encode)
    solutions_t = df.solutions.apply(one_hot_encode)
    
    quizzes_t = torch.cat(quizzes_t.values.tolist())
    solutions_t = torch.cat(solutions_t.values.tolist())
    
    s = df.shape[0]
    randperm = torch.randperm(s)
    train = randperm[: int(train_split * s)]
    test = randperm[int(train_split * s) :]
    return (
        data.TensorDataset(quizzes_t[train], solutions_t[train]),
        data.TensorDataset(quizzes_t[test], solutions_t[test]),
    )


def create_constraint_mask():
    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)

    # row constraints
    for a in range(81):
        r = 9 * (a // 9)
        for b in range(9):
            constraint_mask[a, 0, r + b] = 1

    # column constraints
    for a in range(81):
        c = a % 9
        for b in range(9):
            constraint_mask[a, 1, c + 9 * b] = 1

    # box constraints
    for a in range(81):
        r = a // 9
        c = a % 9
        br = 3 * 9 * (r // 3)
        bc = 3 * (c // 3)
        for b in range(9):
            r = b % 3
            c = 9 * (b // 3)
            constraint_mask[a, 2, br + bc + r + c] = 1
    return constraint_mask


def load_dataset(data_file: str, subsample: int = 10000):
    logger.info("Loading sudoku.csv ...")
    dataset = pd.read_csv(data_file, sep=",")

    my_sample = dataset.sample(subsample)
    train_set, test_set = create_sudoku_tensors(my_sample)

    return train_set, test_set


class SudokuSolver(nn.Module):
    def __init__(self, constraint_mask, n=9, hidden1=100):
	    super(SudokuSolver, self).__init__()
	
	    self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
 	    self.n = n
 	    self.hidden1 = hidden1
 
	    # Feature vector is the 3 constraints
 	    self.input_size = 3 * n
 	    self.l1 = nn.Linear(self.input_size,
 	    self.hidden1, bias=False)
 	    self.a1 = nn.ReLU()
 	    self.l2 = nn.Linear(self.hidden1, n, bias=False)
	    self.softmax = nn.Softmax(dim=1)
	    # x is a (batch, n^2, n) tensor

    def forward(self, x):
 	    n = self.n
	    bts = x.shape[0]
 	    c = self.constraint_mask
 	    min_empty = (x.sum(dim=2) == 0).sum(dim=1).max()
        x_pred = x.clone()

        for a in range(min_empty):
            # score empty numbers
            constraints = (x.view(bts, 1, 1, n * n, n) * c).sum(dim=3)
 
            # empty cells
            empty_mask = (x.sum(dim=2) == 0)
            f = constraints.reshape(bts, n * n, 3 * n)
            y_ = self.l2(self.a1(self.l1(f[empty_mask])))
            s_ = self.softmax(y_)
 
            # Score the rows
            x_pred[empty_mask] = s_
            s = torch.zeros_like(x_pred)
            s[empty_mask] = s_
 
            # find most probable guess
            score, score_pos = s.max(dim=2)
            mmax = score.max(dim=1)[1]
            
            # fill it in
            nz = empty_mask.sum(dim=1).nonzero().view(-1)
            mmax_ = mmax[nz]
            ones = torch.ones(nz.shape[0])
            x.index_put_((nz, mmax_, score_pos[nz, mmax_]), ones)
        return x_pred, x


if __name__ == "__main__":
    load_dataset("%s/data/sudoku_20000.csv" % (os.environ.get("HOME")))
