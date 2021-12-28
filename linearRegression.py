import random
from typing_extensions import Required
import torch
import learningtools.helpplot as plttools
#d21.torch as d21

def synthetic_data(w, b, num_example):
    """生成 y = Xw +b + 噪声。"""
    X = torch.normal(0, 1, ((num_example, len(w))))
    y = torch.matmul(X, w) + b #matmul 高级乘法（维度不统一时会自动广播，将多出的维度看作batch）
    y += torch.normal(0, 0.1, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4,])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0], '\nlabel:',labels[0])

# plttools.set_figsize()
# plttools.plt.scatter(features[:, 1].detach().numpy(),
#                     labels.detach().numpy(), 1)
# plttools.plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

'''init weight'''
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

''' define linear model'''
def linreg(X, w, b):
    return torch.matmul(X, w) + b

'''define loss function'''
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

'''define optimize method'''
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.02
num_epochs = 20
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')

print(f'w的误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的误差：{true_b - b}')
    