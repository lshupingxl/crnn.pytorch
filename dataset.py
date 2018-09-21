#!/usr/bin/python
# encoding: utf-8

import random
import sys
import pathlib
import lmdb
import numpy as np
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import cv2


class ImageDataset(Dataset):
    def __init__(self, data_txt: str, data_shape: tuple, img_channel: int, num_label: int,
                 alphabet: str, transform=None, target_transform=None):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数,应该和网络最终输出的序列宽度一样
        :param alphabet: 字母表
        """
        super(ImageDataset, self).__init__()
        self.data_list = []
        with open(data_txt, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                img_path = pathlib.Path(line[0])
                if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                    self.data_list.append((line[0], line[1]))

        self.transform = transform
        self.target_transform = target_transform
        self.img_h = data_shape[0]
        self.img_w = data_shape[1]
        self.img_channel = img_channel
        self.num_label = num_label
        self.alphabet = alphabet
        self.label_dict = {}
        for i, char in enumerate(self.alphabet):
            self.label_dict[char] = i

    def __getitem__(self, index):
        img_path, label = self.data_list[index]
        label = label.replace(' ', '')
        # try:
        #     label = self.label_enocder(label)
        # except Exception as e:
        #     print(img_path, label)
        img = self.pre_processing(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data_list)

    def label_enocder(self, label):
        """
        对label进行处理，将输入的label字符串转换成在字母表中的索引
        :param label: label字符串
        :return: 索引列表
        """
        tmp_label = np.zeros(self.num_label, dtype=np.float32) -1
        for i, ch in enumerate(label):
            tmp_label[i] = self.label_dict[ch]
        return tmp_label

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img = cv2.imdecode(np.fromfile(img_path), 1 if self.img_channel == 3 else 0)
        img = cv2.resize(img, (110, 16))
        h, w = img.shape[:2]
        ratio_h = self.img_h / h
        new_w = int(w * ratio_h)
        if new_w < self.img_w:
            img = cv2.resize(img, (new_w, self.img_h))
            step = np.zeros((self.img_h, self.img_w - new_w, self.img_channel), dtype=img.dtype)
            img = img.reshape((img.shape[0],img.shape[1],self.img_channel))
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (self.img_w, self.img_h))
        return img


class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()), 'utf-8')

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
