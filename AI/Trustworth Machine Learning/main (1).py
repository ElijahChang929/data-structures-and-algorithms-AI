from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
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
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy


# 下面导入的类是很早版本的，现在版本已经没有了
# from torch.autograd.gradcheck import zero_gradients
from train import test_loader, adver_nums, device, batch_size, simple_model


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=100):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        # print("Using GPU")
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            if x.grad is not None:
                x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop_i, label, k_i, pert_image


# 这几个变量主要用于之后的测试以及可视化
adver_example_by_FOOL = torch.zeros((batch_size, 1, 28, 28)).to(device)
adver_target = torch.zeros(batch_size).to(device)
clean_example = torch.zeros((batch_size, 1, 28, 28)).to(device)
clean_target = torch.zeros(batch_size).to(device)
# 从test_loader中选取1000个干净样本，使用deepfool来生成对抗样本
for i, (data, target) in enumerate(test_loader):
    if i >= adver_nums / batch_size:
        break
    if i == 0:
        clean_example = data
    else:
        clean_example = torch.cat((clean_example, data), dim=0)

    cur_adver_example_by_FOOL = torch.zeros_like(data).to(device)

    for j in range(batch_size):
        r_rot, loop_i, label, k_i, pert_image = deepfool(data[j], simple_model)
        cur_adver_example_by_FOOL[j] = pert_image

    # 使用对抗样本攻击VGG模型
    pred = simple_model(cur_adver_example_by_FOOL).max(1)[1]
    # print (simple_model(cur_adver_example_by_FOOL).max(1)[1])
    if i == 0:
        adver_example_by_FOOL = cur_adver_example_by_FOOL
        clean_target = target
        adver_target = pred
    else:
        adver_example_by_FOOL = torch.cat((adver_example_by_FOOL, cur_adver_example_by_FOOL), dim=0)
        clean_target = torch.cat((clean_target, target), dim=0)
        adver_target = torch.cat((adver_target, pred), dim=0)

print(adver_example_by_FOOL.shape)
print(adver_target.shape)
print(clean_example.shape)
print(clean_target.shape)

import torch
import torch.utils.data as Data
from tqdm import tqdm

from deepfool_method import adver_example_by_FOOL, clean_target
from train import simple_model, adver_nums, device, batch_size


def adver_attack_vgg(model, adver_example, target, name):
    adver_dataset = Data.TensorDataset(adver_example, target)
    loader = Data.DataLoader(
        dataset=adver_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size  # 每块的大小
    )
    correct_num = torch.tensor(0).to(device)
    for j, (data, target) in tqdm(enumerate(loader)):
        data = data.to(device)
        target = target.to(device)
        pred = model.forward(data).max(1)[1]
        num = torch.sum(pred == target)
        correct_num = correct_num + num
    print(correct_num)
    print('\n{} correct rate is {}'.format(name, correct_num / adver_nums))


adver_attack_vgg(simple_model, adver_example_by_FOOL, clean_target, 'simple model')
from matplotlib import pyplot as plt
from deepfool_method import adver_example_by_FOOL, adver_target, clean_target, clean_example


def plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target):
    n_cols = 5
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(n_cols*4,n_rows*2))
    for i in range(n_cols):
        for j in range(n_rows):
            plt.subplot(n_cols,n_rows*2,cnt1)
            plt.xticks([])
            plt.yticks([])
            plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(clean_example[cnt].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
            plt.subplot(n_cols, n_rows*2,cnt1+1)
            plt.xticks([])
            plt.yticks([])
            # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(adver_example[cnt].reshape(28, 28).to('cpu').detach().numpy(), cmap='gray')
            cnt = cnt + 1
            cnt1 = cnt1 + 2
    plt.show()
plot_clean_and_adver(adver_example_by_FOOL, adver_target, clean_example, clean_target)
