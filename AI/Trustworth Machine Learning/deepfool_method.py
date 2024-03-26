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

