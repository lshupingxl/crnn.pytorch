from __future__ import print_function
import argparse
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torch.utils.data
import os
import utils
import dataset
from models import crnn
from data_generator.config import Alphabet
import pandas
import time
from tensorboardX import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
gpu_id = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print('train with gpu %s and pytorch %s' % (gpu_id, torch.__version__))


def val(net, test_loader, criterion, converter, device):
    net.eval()
    n_correct = 0
    val_loss = 0.0

    for i, (images, labels) in enumerate(test_loader):
        batch_size = images.size(0)
        text, length = converter.encode(labels)
        text = torch.Tensor(text).int()
        length = torch.Tensor(length).int()
        images = images.to(device)

        preds = net(images)
        preds_size = torch.Tensor([preds.size(0)] * batch_size).int()
        loss = criterion(preds, text, preds_size, length)
        val_loss += loss.item()

        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.cpu().data, preds_size.cpu().data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.detach().data, preds_size.detach().data, raw=True)[:opt.n_test_disp]
    pandas_show = [[raw_pred, pred, gt] for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels)]
    print(pandas.DataFrame(data=pandas_show, columns=['network_output', 'ctc_output', 'ground_truth']))

    accuracy = n_correct / float(test_loader.dataset.__len__())
    print('Test loss: %f, accuray: %f' % (val_loss / test_loader.dataset.__len__(), accuracy))
    return accuracy, val_loss / test_loader.dataset.__len__()


def train(opt):
    if opt.output_dir is None:
        opt.output_dir = 'output/'
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    nc = 1
    device = torch.device("cuda:0" if gpu_id is not None and torch.cuda.is_available() else "cpu")
    # ************************* image dataset
    # train_dataset = dataset.ImageDataset(data_txt=opt.trainfile, data_shape=(opt.imgH, opt.imgW), img_channel=nc,
    #                                      num_label=80, alphabet=opt.alphabet, transform=transforms.ToTensor())
    # test_dataset = dataset.ImageDataset(data_txt=opt.valfile, data_shape=(opt.imgH, opt.imgW), img_channel=nc,
    #                                     num_label=80, alphabet=opt.alphabet, transform=transforms.ToTensor())
    #
    train_transform = transforms.Compose([transforms.Resize((opt.imgH, opt.imgW)), transforms.ToTensor()])
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, transform=train_transform)
    test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=int(opt.workers))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True,
                                              num_workers=int(opt.workers))

    # ************************* origin dataset and dataloader
    # train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
    #                                            num_workers=int(opt.workers),
    #                                            collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW,
    #                                                                            keep_ratio=opt.keep_ratio))
    # test_dataset = dataset.lmdbDataset(root=opt.valroot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=opt.batchSize,
    #                                           num_workers=int(opt.workers))
    converter = utils.strLabelConverter(opt.alphabet)

    net = crnn.CRNN(opt.imgH, nc, len(opt.alphabet), opt.nh).to(device)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        net.load_state_dict(torch.load(opt.crnn))

    net = net.to(device)
    # writer = SummaryWriter('./log/%s' % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
    writer = SummaryWriter(opt.output_dir)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, nc, opt.imgH, opt.imgW).to(device))
    # writer.add_graph(model=net, input_to_model=dummy_input)

    criterion = nn.CTCLoss(reduction='sum')
    criterion = criterion.to(device)

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    for epoch in range(opt.epochs):
        net.train()
        # Adjust lr
        scheduler.step()
        start = time.time()
        cur_step = 0
        for i, (images, labels) in enumerate(train_loader):
            n_correct = 0
            batch_size = images.size(0)
            text, length = converter.encode(labels)

            text = torch.Tensor(text).int()
            length = torch.Tensor(length).int()
            images = images.to(device)

            preds = net(images)
            preds_size = torch.Tensor([preds.size(0)] * batch_size).int()
            preds.requires_grad_(True)
            loss = criterion(preds, text, preds_size, length)  # text,preds_size must be cpu
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = preds.max(2)
            preds = preds.squeeze(1)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.cpu().data, preds_size.cpu().data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1
            # write tensorboard
            cur_step = epoch * (train_dataset.__len__() / batch_size) + i
            writer.add_scalar(tag='Train/loss', scalar_value=loss.item() / batch_size, global_step=cur_step)
            writer.add_scalar(tag='Train/acc', scalar_value=n_correct / batch_size, global_step=cur_step)
            writer.add_scalar(tag='Train/lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)
            # display msg
            if (i + 1) % opt.displayInterval == 0:
                batch_time = time.time() - start
                start = time.time()
                print('[%d/%d][%d/%d] Loss:%f Acc:%f Time:%fs Lr:%f' % (
                    epoch, opt.epochs, (i + 1), len(train_loader), loss / batch_size, n_correct / batch_size,
                    batch_time,
                    scheduler.get_lr()[0]))
        # test
        val_acc, val_loss = val(net, test_loader, criterion, converter, device)
        # write tensorboard
        writer.add_scalar(tag='Eval/acc', scalar_value=val_acc, global_step=cur_step)
        writer.add_scalar(tag='Eval/loss', scalar_value=val_loss, global_step=cur_step)
        # save params
        torch.save(net.state_dict(), '{0}/netCRNN_{1}.pth'.format(opt.output_dir, epoch))
    # save final model
    torch.save(net, opt.output_dir + '/model.pkl')
    torch.onnx.export()
    writer.close()


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainroot', default='/data2/zj/data/num//train_lmdb', help='path to dataset')
    parser.add_argument(
        '--valroot', default='/data2/zj/data/num/test_lmdb', help='path to dataset')
    parser.add_argument('--trainfile', default='/data2/zj/data/crnn/all/Chinese_character/train2.txt', help='path to dataset file')
    parser.add_argument('--valfile', default='/data2/zj/data/crnn/all/Chinese_character/test2.txt', help='path to dataset file')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=24)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=320, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default=Alphabet.CHINESECHAR_LETTERS_DIGIT_SYMBOLS)
    parser.add_argument('--output_dir', default='output/output_gru_default1', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=1, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--keep_ratio', default=True, action='store_true',
                        help='whether to keep ratio for image resize')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = init_args()
    print(opt)
    train(opt)
