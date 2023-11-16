"""

# Link to PyTorch中的 数独 RNN
https://www.toutiao.com/a6640028992575373828/?timestamp=1586213213&app=news_article&group_id=6640028992575373828&req_id=202004070646530100140470310D689AE5
"""
import json
import os
import pandas as pd
from pypchutils.generic import create_logger
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

logger = create_logger(__name__)


def create_sudoku_tensors(df: pd.DataFrame, train_split: float = 0.5):
    def one_hot_encode(x: str):
        zeros = torch.zeros((1, 81, 9), dtype=torch.float)
        for m in range(81):
            # -1 because
            zeros[0, m, int(x[m]) - 1] = 1 if int(x[m]) > 0 else 0
        return zeros

    quizzes_t = df.quizzes.apply(one_hot_encode)
    solutions_t = df.solutions.apply(one_hot_encode)

    quizzes_t = torch.cat(quizzes_t.values.tolist())
    solutions_t = torch.cat(solutions_t.values.tolist())

    x = df.shape[0]
    randperm = torch.randperm(x)
    train = randperm[: int(train_split * x)]
    test = randperm[int(train_split * x) :]
    return (
        data.TensorDataset(quizzes_t[train], solutions_t[train]),
        data.TensorDataset(quizzes_t[test], solutions_t[test]),
    )


def create_constraint_mask():
    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)

    # row constraints
    for m in range(81):
        r = 9 * (m // 9)
        for b in range(9):
            constraint_mask[m, 0, r + b] = 1

    # column constraints
    for m in range(81):
        c = m % 9
        for b in range(9):
            constraint_mask[m, 1, c + 9 * b] = 1

    # box constraints
    for m in range(81):
        r = m // 9
        c = m % 9
        br = 3 * 9 * (r // 3)
        bc = 3 * (c // 3)
        for b in range(9):
            r = b % 3
            c = 9 * (b // 3)
            constraint_mask[m, 2, br + bc + r + c] = 1

    logger.info("Successfully completed create_constraint_mask()")
    return constraint_mask


def load_dataset(data_file: str, subsample_pct: float = 0.5):
    logger.info(f"Loading data from {data_file}...")
    dataset = pd.read_csv(data_file, sep=",")

    samples = dataset.sample(int(len(dataset) * subsample_pct))
    train_set, test_set = create_sudoku_tensors(samples)
    return train_set, test_set


class SudokuSolver(nn.Module):
    def __init__(self, constraint_mask, n: int = 9, hidden1: int = 100):
        super(SudokuSolver, self).__init__()

        self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
        self.n = n
        self.hidden1 = hidden1

        # Feature vector is the 3 constraints
        self.input_size = 3 * n
        self.l1 = nn.Linear(self.input_size, self.hidden1, bias=False)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden1, n, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, verbose: int = 1):
        """
        Args:
            x: A tensor of (batch, n^2, n)
            verbose (int, optional)
        """
        n = self.n
        bts = x.shape[0]  # batch_size
        c = self.constraint_mask
        min_empty = (x.sum(dim=2) == 0).sum(dim=1).max()
        x_pred = x.clone()

        if verbose >= 2:
            logger.info(f"foward(): batch size: {bts}")

        for _ in range(min_empty):
            # score empty numbers
            constraints = (x.view(bts, 1, 1, n * n, n) * c).sum(dim=3)

            # empty cells
            empty_mask = x.sum(dim=2) == 0
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


def main(
    data_file: str = "%s/Google Drive/xinheng/data/sudoku_20000.csv" % (os.environ.get("HOME")),
    batch_size: int = 100,
    epochs: int = 20,
    learning_rate=0.01,
):
    train_set, test_set = load_dataset(data_file=data_file, subsample_pct=0.5)
    constraint_mask = create_constraint_mask()

    dataloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # dataloader_val = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    loss = nn.MSELoss()
    sudoku_solver = SudokuSolver(constraint_mask)
    optimizer = optim.Adam(sudoku_solver.parameters(), lr=learning_rate, weight_decay=0.000)

    loss_train, loss_val = [], []
    logger.info(f"Trained has started. epochs: {epochs}, batch_size: {batch_size}")
    for epoch_idx in range(epochs):
        for batch_idx, ts in enumerate(dataloader):
            #
            sudoku_solver.train()
            optimizer.zero_grad()
            pred, mat = sudoku_solver(ts[0])
            ls = loss(pred, ts[1])
            ls.backward()
            optimizer.step()

            logger.info("Epoch " + str(epoch_idx) + " batch " + str(batch_idx) + ": " + str(ls.item()))
            sudoku_solver.eval()
            with torch.no_grad():
                n = 100
                rows = torch.randperm(test_set.tensors[0].shape[0])[:n]
                test_pred, test_fill = sudoku_solver(test_set.tensors[0][rows])
                errors = test_fill.max(dim=2)[1] != test_set.tensors[1][rows].max(dim=2)[1]
                loss_val.append(errors.sum().item())
                logger.info("Cells in error: " + str(errors.sum().item()))
    return sudoku_solver


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default=f"{os.environ.get('HOME')}/Google Drive/xheng/data/sudoku_20000.csv",
        help=f"{os.environ.get('HOME')}/data/sudoku_20000.csv,"
        f"{os.environ.get('HOME')}/Google Drive/xheng/data/sudoku_20000.csv",
    )

    # Parse the cmd line args
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
