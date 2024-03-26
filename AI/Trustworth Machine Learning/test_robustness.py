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
