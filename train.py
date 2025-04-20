import os
from PIL import Image
import random
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import math
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import SimpleITK as sitk
from tensorboardX import SummaryWriter
from dual_dis_topk import dualdis_vits_patch16_224
from losses.cls_loss import LabelSmoothCELoss, BCE_LOSS, SoftmaxEQLV2Loss, SoftTargetCrossEntropy
from torchvision import transforms
from transform import video_transforms, volume_transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target)
        pred = torch.softmax(pred,dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)
        log_p = probs.log()

        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha

        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class KZDataset():
    def __init__(self, path_0=None,path_1=None, typ='train',transform=None, rand=False):
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)
        if rand:
            random.seed(1)
            random.shuffle(self.data_info_0)
            random.shuffle(self.data_info_1)

        if typ == 'val':
            self.data_info = self.data_info_0 + self.data_info_1
            self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(8)])  # (C, T, H, W)
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(256, interpolation='bilinear'),   # (T, H, W, C)
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif typ == 'train':
            self.data_info = self.data_info_0 + self.data_info_1
            self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(8)])  # (C, T, H, W)
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(256, interpolation='bilinear'),  # (T, H, W, C)
                video_transforms.RandomCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        print(len(self.data_info))

        # print(self.data_info)
        # print('0_length:',len(self.data_info_0),'1_length:',len(self.data_info_1))
        # print(len(self.data_info))
        self.is_transf = transform

    def __getitem__(self, index):
    	# Dataset读取图片的函数
        img_pth, label = self.data_info[index]
        img = sitk.ReadImage(img_pth)
        img = sitk.GetArrayFromImage(img)  #32,224,224,3
        patient = img_pth.split('\\')[-1][:-4]
        if self.is_transf:
            img =  torch.tensor(img).permute(3, 0, 1, 2)  # 3,32,224,224
            img = self.data_samplr(img)  # 3,8,224,224
            img =  np.array(img.permute(1, 2, 3, 0))  # 8,224,224,3
            img = self.data_transform(img)
        img = img.transpose(0, 1) #8,3,224,224
        return img, label, patient
    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
        data_info = []
        data = open(txt_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.replace(",", " ").replace("\n", "")
            data_line = data_line.split()
            img_pth = data_line[0]
            label = int(data_line[1])
            data_info.append((img_pth, label))
        return data_info   
    
    
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    os.makedirs('./' + store_name, exist_ok=True)
    exp_dir = './' + store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    K = 1
    for ki in range(0,K):
        use_cuda = torch.cuda.is_available()
        print(use_cuda)

        log_dir = './' + store_name +'/'+'logs'
        writer = SummaryWriter(log_dir)

        print('==> Preparing data..')

        trainset = KZDataset(path_0=r'your_train_data_0.csv',
                             path_1=r'your_train_data_1.csv',
                             typ='train', transform=True, rand=False)
        valset = KZDataset(path_0=r'your_train_data_0.csv',
                           path_1=r'your_train_data_1.csv',
                           typ='val', transform=True, rand=False)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        if resume:
            v = torch.load(model_path)
        else:
            v = dualdis_vits_patch16_224(pretrained=True)
        premodel_video = torch.load(
            './pre_weight/model_bce.pth',
            map_location="cpu")
        checkpoint_i = torch.load('./distillation_weight/dist_image.pth',
                                  map_location="cpu")
        checkpoint_v = premodel_video.state_dict()
        state_dict_v = checkpoint_v

        state_dict_i = checkpoint_i
        for old_key in list(state_dict_i.keys()):
            if old_key.startswith('image_encoder.'):
                new_key_i = old_key[14:]
                state_dict_i[new_key_i] = state_dict_i.pop(old_key)
            else:
                del state_dict_i[old_key]

        state_dict_vi = OrderedDict()
        for key_v in state_dict_v:
            new_key_v = 'vdmodel.' + key_v
            state_dict_vi[new_key_v] = state_dict_v[key_v]
        for key_i in state_dict_i:
            new_key_i = 'immodel.' + key_i
            state_dict_vi[new_key_i] = state_dict_i[key_i]

        for key_head in ['head.weight', 'head.bias']:
            state_dict_vi[key_head] = state_dict_v[key_head]

        for key_vi_head in ['vdmodel.head.weight', 'vdmodel.head.bias', 'immodel.neck.0.weight',
                            'immodel.neck.1.weight',
                            'immodel.neck.1.bias', 'immodel.neck.2.weight', 'immodel.neck.3.weight',
                            'immodel.neck.3.bias']:
            del state_dict_vi[key_vi_head]

        state_dict = state_dict_vi
        v.load_state_dict(state_dict,
                          strict=False)  # strict=True, Missing key(s) in state_dict: "norm.weight", "norm.bias".
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v.to(device)

        # CELoss = FocalLoss()
        # CELoss = nn.CrossEntropyLoss()
        CELoss = BCE_LOSS()
        # CELoss = SoftmaxEQLLoss(num_classes=2)
        # CELoss = SoftmaxEQLV2Loss(num_classes=2)
        # CELoss = SoftTargetCrossEntropy()

        optimizer = optim.SGD([{'params': v.parameters(), 'lr': 0.001}], momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.AdamW(v.parameters(), lr=0.001, betas=(0.5, 0.9) )

        max_val_acc = 0
        max_val_acc_e = 0
        lr = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001]
        # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
        # lr = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.00002]
        # lr = [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.0004]
        for epoch in range(start_epoch, nb_epoch):
            #print('\nEpoch: %d' % epoch)
            v.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets, patient) in enumerate(trainloader):
                inputs = inputs.permute(0, 2, 1, 3, 4).float()  # BTCHW->BCTHW
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                'loss'
                # optimizer.zero_grad()
                # output_concat = v(inputs)
                # concat_loss = CELoss(output_concat, targets)
                # concat_loss.backward()
                # optimizer.step()
                'loss'
                'adv_loss'
                output_concat = v(inputs,3)
                concat_loss = CELoss(output_concat, targets)
                b = torch.tensor(0.008)
                flood = (concat_loss-b).abs() + b
                optimizer.zero_grad()
                flood.backward()
                optimizer.step()
                'adv_loss'

                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                train_loss += concat_loss.item()

                if batch_idx % 5 == 0:
                    # viz.line([[train_loss / (batch_idx + 1), 100. * float(correct) / total]],
                    #          [[epoch*400+batch_idx, epoch*400+batch_idx]], win='train_loss', update='append')
                    print(
                        'K-fold %d, Epoch %d, Step: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        ki, epoch, batch_idx, train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (batch_idx + 1)

            writer.add_scalar('train/train_loss', train_loss, epoch)
            writer.add_scalar('train/train_acc', train_acc, epoch)
            with open(exp_dir + '/results_train_np_%d.txt' % ki, 'a') as file:
                file.write(
                    'K-fold %d, Epoch %d | train_acc = %.5f | train_loss = %.5f\n' % (
                    ki, epoch, train_acc, train_loss))

            torch.cuda.empty_cache()


            # test ------------------------------------------
            testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
            v.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets, patient) in enumerate(testloader):
                inputs = inputs.permute(0, 2, 1, 3, 4).float()  # BTCHW->BCTHW
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                with torch.no_grad():
                    output = v(inputs,3)
                loss = CELoss(output, targets)
        
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                # print('label:',targets.data.cpu(),'pred:',predicted.cpu())
        
                if batch_idx % 50 == 0:
                    print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        
            test_acc = 100. * float(correct) / total
            test_loss = test_loss / (batch_idx + 1)

            writer.add_scalar('test/test_loss', test_loss, epoch)
            writer.add_scalar('test/test_acc', test_acc, epoch)
            with open(exp_dir + '/results_test_np_%d.txt' % ki, 'a') as file:
                file.write(
                    'K-fold %d, Epoch %d | test_acc = %.5f | test_loss = %.5f\n' % (
                    ki, epoch, test_acc, test_loss))


            if test_acc > max_val_acc: # and epoch > int(nb_epoch/2):
                print('Best test accuracy, %f' % test_acc)
                max_val_acc = test_acc
                v.cpu()
                torch.save(v, './TOP3/tvt811RJMC/' + store_name + '/model_bce_5fold%d.pth'% ki)
                # if test_acc > max_val_acc_e and epoch > int(nb_epoch/2):
                #     max_val_acc_e = test_acc
                #     torch.save(v, './' + store_name + '/e_model_focal_5fold%d.pth' % ki)
                v.to(device)

        torch.cuda.empty_cache()

train(nb_epoch=50,             # number of epoch
         batch_size=2,         # batch size
         store_name='VAT_bce_sgd_0.001_e50', #'RJIndp_5f_bce_sgd_0.002_e100',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path=None)         # the saved model where you want to resume the trai