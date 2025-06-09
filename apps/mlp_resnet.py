import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim),
                      norm(dim))
    f = nn.Sequential(nn.Residual(fn), nn.ReLU())
    return f
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    f = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
    for i in range(num_blocks):
        f = nn.Sequential(f, ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    f = nn.Sequential(f, nn.Linear(hidden_dim, num_classes))
    return f
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    softmax_loss = nn.SoftmaxLoss()
    errors, loss_val = [], 0
    for i, batch in enumerate(dataloader):
        x, y = batch
        h = model(x)
        loss = softmax_loss(h, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        loss_val += loss.numpy() * x.shape[0]
        errors += (h.numpy().argmax(axis=1) != y.numpy()).tolist()

    return np.mean(np.array(errors)), loss_val / len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"), os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )
    mnist_dataloader = ndl.data.DataLoader(
        dataset=mnist_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_err, train_loss, test_err, test_loss = 0, 0, 0, 0

    for i in range(epochs):
        train_err, train_loss = epoch(mnist_dataloader, model, opt)
        if i == epochs - 1:
            test_err, test_loss = epoch(test_dataloader, model, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
