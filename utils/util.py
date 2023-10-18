import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch_directml
import matplotlib as ma
import matplotlib.pyplot as plt
from IPython import display
dml = torch_directml.device()
import time


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def set_axes(ax, xlabel=None, ylabel=None, xlim=None, ylim=None, 
             xscale=None, yscale=None, legend=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if legend:
        ax.legend(legend)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, 
                 ylim=None, xscale='linear', yscale='linear', 
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=250)
        if nrows * ncols == 1:
            self.axes = [self.axes, ] #将单个变量变成列表，便于统一操作
        self.config_axes =  lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        # 默认支持最多绘制 4 条线
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.axes[0].grid()
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def show_images(imgs, row, col, scales = 3, titles = None):
    fig, axes = plt.subplots(row, col, figsize=(col * scales, row * scales))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy()
        ax.imshow(img)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.show()

def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred  for true, pred in zip(trues, preds)]
    show_images(X[0:n].view(-1, 28, 28), 1, n, titles=titles[0:n])

def plot(x, y, xlabel=None, ylabel=None, legend=None, 
         figsize=(6,3), dpi=250, xlim=None, ylim = None, title=None):
    """可以绘制多个曲线, 但是请将每个曲线放置在嵌套列表的里面"""
    if not hasattr(y[0], "__len__") or torch.is_tensor(y[0]):
        y, x = [y], [x]
    n = len(y)
    X = [[] for _ in range(n)]
    Y = [[] for _ in range(n)]

    for i, (x_, y_) in enumerate(zip(x, y)):
        for xval, yval in zip(x_, y_):
            if torch.is_tensor(xval) or torch.is_tensor(yval):
                X[i].append(xval.item())
                Y[i].append(yval.item())
            else:
                X[i].append(xval)
                Y[i].append(yval)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.grid()
    ax.set_title(title)
    for i, (xdata, ydata) in enumerate(zip(X, Y)):
        ax.plot(xdata, ydata)
    set_axes(ax, xlabel, ylabel, xlim, ylim, legend=legend)
    plt.show()
    
if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    plot([1, 2, 3], [1, 2, 3], 'x', 'relu(x)', figsize=(5, 2.5))