from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# 加载mnist数据集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=10, shuffle=True)

# 超参数设置
batch_size = 10
epoch = 1
learning_rate = 0.001
# 生成对抗样本的个数
adver_nums = 1000


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 选择设备
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# 初始化网络，并定义优化器
simple_model = Net().to(device)
optimizer1 = torch.optim.SGD(simple_model.parameters(), lr=learning_rate, momentum=0.9)
print(simple_model)


# 训练模型
def train(model, optimizer):
    for i in range(epoch):
        for j, (data, target) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            target = target.to(device)
            logit = model(data)
            loss = F.nll_loss(logit, target)
            model.zero_grad()
            # 如下：因为其中的loss是单个tensor就不能用加上一个tensor的维度限制
            loss.backward()
            # 如下有两种你形式表达，一种是原生，一种是使用optim优化函数直接更新参数
            # 为什么原生的训练方式没有效果？？？代表参数没有更新，就离谱。
            # 下面的detach与requires_grad_有讲究哦，终于明白了；但是为什么下面代码不能work还是没搞懂
            # for params in model.parameters():
            #   params = (params - learning_rate * params.grad).detach().requires_grad_()
            optimizer.step()
            if j % 1000 == 0:
                print('第{}个数据，loss值等于{}'.format(j, loss))


train(simple_model, optimizer1)

# eval eval ，老子被你害惨了
# 训练完模型后，要加上，固定DROPOUT层
simple_model.eval()


# 模型测试
def m_test(model, name):
    correct_num = torch.tensor(0).to(device)
    for j, (data, target) in tqdm(enumerate(test_loader)):
        data = data.to(device)
        target = target.to(device)
        logit = model(data)
        pred = logit.max(1)[1]
        num = torch.sum(pred == target)
        correct_num = correct_num + num
        print(correct_num)
        print('\n{} correct rate is {}'.format(name, correct_num/10000))


m_test(simple_model, 'simple model')
