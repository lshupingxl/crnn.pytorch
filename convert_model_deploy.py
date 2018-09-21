# -*- coding: utf-8 -*-
# @Time    : 18-9-21 上午11:42
# @Author  : zhoujun
import torch
from models.crnn import CRNN


def save(net, input, save_path):
    net.eval()
    traced_script_module = torch.jit.trace(net, input)
    traced_script_module.save(save_path)

def load(model_path):
    return torch.jit.load(model_path)

if __name__ == '__main__':
    input = torch.Tensor(10, 3, 32, 320)
    model_path = './model.pth'
    net = CRNN(32, 3, 10, 256)
    net.load_state_dict(torch.load(model_path))
    save(net, input, './model.pt')
