"""
# Links:
https://towardsdatascience.com/how-to-deploy-your-pytorch-models-with-torchserve-2452163871d3
https://github.com/pytorch/serve/blob/master/README.md

"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from AI.gist.torchserve.model import Net


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main(
    epochs: int,
    batch_size: int,
    test_batch_size: int,
    lr: float,  # learning rate
    gamma,
    log_interval: int,
    no_cuda: bool,
    save_model: bool,
    seed: int,
):
    torch.manual_seed(seed)

    use_cuda = not no_cuda and torch.cuda.is_available()
    print(f"use_cuda is {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    # Do it
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = vars(parser.parse_args())
    print("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
