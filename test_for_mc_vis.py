import os
import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from vis.visualizer import get_local
get_local.activate()
import SimpleITK as sitk
from PIL import Image, ImageDraw
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,accuracy_score
import numpy as np
import math
import json
from scipy.io import savemat
import matplotlib.pyplot as plt
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


def visualize_grid_to_grid_con(patient, att_maps, image, grid_size=28, ly=11, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)  # 28,28
    att_map = att_maps[0]
    H, W = att_map.shape # 784，784
    with_cls_token = False
    mask_list = []
    image_np = image[0, 5].numpy()  # tensor: 224,224,3
    single_image = Image.fromarray(np.uint8(image_np))  # image: 512*256
    for grid_index in range(196):
        mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
        mask_list.append(mask)
    mask_arr = np.array(mask_list)
    mask_arr = np.mean(mask_arr, axis=0)
    mask_im = Image.fromarray(mask_arr).resize((single_image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    # ax[1].imshow(grid_image)
    ax[0].imshow(mask_im / np.max(mask_im), alpha=1, cmap='rainbow')
    ax[0].axis('off')
    plt.savefig('./attnmap/AHAQ/' + patient + 'L'+ str(ly)+'_AH_AQ.png')
    #plt.show()
    plt.close(fig)


def visualize_grid_to_grid_avg(patient, att_maps, grid_index, image, grid_size=28, ly=11, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)  # 28,28
    att_map = att_maps[0]
    H, W = att_map.shape # 784，784
    with_cls_token = False

    image_np = image[0, 5].numpy()  # tensor: 224,224,3
    single_image = Image.fromarray(np.uint8(image_np))  # image: 512*256

    grid_image = highlight_grid(single_image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((single_image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')
    # ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=1, cmap='rainbow')
    ax[1].axis('off')
    plt.savefig('./attnmap/AH/' + patient + 'L'+ str(ly)+'_AH_' + 'Q' + str(grid_index) + '.png')
    #plt.show()
    plt.close(fig)

# 784=4*14*14=28*28
def visualize_grid_to_grid(patient, att_maps, head_index, grid_index, image, grid_size=28, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)  # 28,28
    att_map = att_maps[0, head_index, :, :]
    H, W = att_map.shape # 784，784
    with_cls_token = False

    image_np = image[0, 5].numpy()  # tensor: 224,224,3
    single_image = Image.fromarray(np.uint8(image_np))  # image: 512*256

    grid_image = highlight_grid(single_image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((single_image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')
    # ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=1, cmap='rainbow')
    ax[1].axis('off')
    plt.savefig('./attnmap/' + patient + 'L11H' + str(head_index) + 'Q' + str(grid_index) + '.png')
    #plt.show()


def highlight_grid(image, grid_indexes, grid_size=28):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, 2*grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image


def visualize_combined_attention_map(attention_maps, image, alpha=0.6):
    #  (batch_size, num_heads, height, width)
    combined_att_map = np.mean(attention_maps, axis=1)[0]  # head_avg: 1,6,784,784->1,784,784
    #
    att_map_normalized = (combined_att_map - np.min(combined_att_map)) / (
            np.max(combined_att_map) - np.min(combined_att_map))  # norm [0, 1]

    att_map_colored = plt.cm.rainbow(att_map_normalized)  # (784, 784, 4)
    att_map_colored = (att_map_colored[:, :, :3] * 255).astype(np.uint8)  #

    plt.figure(figsize=(10, 7))
    # plt.imshow(combined_image)
    plt.imshow(att_map_colored)
    plt.axis('off')
    plt.show()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class KZDataset_test():
    def __init__(self, path_0=None,path_1=None,transform=None, rand=False):
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)
        self.data_info = self.data_info_0 + self.data_info_1
        print(len(self.data_info))
        if rand:
	        random.seed(1)
        	random.shuffle(self.data_info)
        # print(self.data_info)
        # print('0_length:',len(self.data_info_0),'1_length:',len(self.data_info_1))
        # print(len(self.data_info))

        self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(8)])  # (C, T, H, W)
        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(256, interpolation='bilinear'),  # (T, H, W, C)
            video_transforms.CenterCrop(size=(224, 224)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.is_transf = transform

    def __getitem__(self, index):
        img_pth, label = self.data_info[index]
        img = sitk.ReadImage(img_pth)
        img = sitk.GetArrayFromImage(img)  # 32,224,224,3
        patient = img_pth.split('\\')[-1][:-4]
        if self.is_transf:
            img = torch.tensor(img).permute(3, 0, 1, 2)  # 3,32,224,224
            img = self.data_samplr(img)  # 3,8,224,224
            img = np.array(img.permute(1, 2, 3, 0))  # 8,224,224,3
            img = self.data_transform(img)
        img = img.transpose(0, 1)  # 8,3,224,224
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


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


mark = 'CV'
if mark == 'CV':
    K =5
    ff_lab = []
    ff_pre = []
    ff_pre_nor = []
    ff_name = []
    func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
    for ki in range(0,1):
        model_path = r'your_model.pth'
        v = torch.load(model_path)
        v.eval()
        use_cuda = torch.cuda.is_available()
        test_loss = 0
        correct = 0
        correct_com = 0
        total = 0
        idx = 0
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        
        v.to(device)

        transform_img = transforms.Compose([
            # transforms.CenterCrop(800),
            # transforms.Resize(286),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        testset = KZDataset(path_0=r'your_train_data_0.csv',
                            path_1=r'your_train_data_1.csv',
                            typ='val', transform=transform_img, rand=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        final_lab = []
        final_pre = []
        final_pre_nor = []
        final_name = []
        for batch_idx, (inputs, targets,patient) in enumerate(testloader):
            idx = batch_idx
            inputs0 = inputs.permute(0, 1, 3, 4, 2).float()  # 1.8.224.224.3
            inputs = inputs.permute(0, 2, 1, 3, 4).float()
            inputs, targets = inputs.to(device), targets.to(device)
            output_concat = v(inputs, 3)

            ### get attention map
            cache = get_local.cache
            print(list(cache.keys()))
            attention_maps = cache['Attention.forward']
            print(len(attention_maps))  # 12  depth=12
            ### There are 12 heads each layer
            for l in range(12):
                attn_map_list = []
                print(attention_maps[l].shape)  # 1,6,784,784  heads=6
                avg_attn_map = np.mean(attention_maps[l], axis=1)  # 1,784,784
                avg_attn_map1 = avg_attn_map[:,:196,:196]
                attn_map_list.append(avg_attn_map1)

                avg_attn_map2 = avg_attn_map[:,196:392,196:392]
                attn_map_list.append(avg_attn_map2)

                avg_attn_map3 = avg_attn_map[:,392:588,392:588]
                attn_map_list.append(avg_attn_map3)

                avg_attn_map4 = avg_attn_map[:,588:,588:]
                attn_map_list.append(avg_attn_map4)

                attn_map_list = np.array(attn_map_list)
                attn_map_mean = np.mean(attn_map_list, axis=0)

                visualize_grid_to_grid_con(patient[0], attn_map_mean, inputs0, grid_size=14, ly=l)
                visualize_grid_to_grid_avg(patient[0], attn_map_mean, 90, inputs0, grid_size=14, ly=l)
            # for n in range(64):  # 784=4*14*14=28*28, 224/28=8, 8*8=64
            #     visualize_grid_to_grid_avg(patient[0], avg_attn_map, n, inputs0, grid_size=28)  # layer24,head12
            # # for k in range(6):
            #     for n in range(64):  # 784=4*14*14=28*28, 224/28=8, 8*8=64
            #         visualize_grid_to_grid_28(patient[0], attention_maps[11], k, n, inputs0, grid_size=28)  # layer24,head12
                get_local.clear()
                continue
else:

    func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
    model_path=r'your_model.pth'
    net = torch.load(model_path)
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    batch_size = 1
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    net.to(device)

    idxs = [['B', 'B0', 'B1'], ['C', 'C0', 'C1'], ['D', 'D0', 'D1'], ['E', 'E0', 'E1'],
            ['F', 'F0', 'F1'], ['G', 'G0', 'G1'], ['H', 'H0', 'H1'], ['I', 'I0', 'I1'],
            ['J', 'J0', 'J1'], ['K', 'K0', 'K1'], ['L', 'L0', 'L1'], ['M', 'M0', 'M1'],
            ['N', 'N0', 'N1'], ['O', 'O0', 'O1'], ['Main', 'test0', 'test1']]
    for idx_i in idxs:
        testset = KZDataset_test(path_0=r'your_test_path_0\%s.csv' % idx_i[1],
                                 path_1=r'your_test_path_1\%s.csv' % idx_i[2],
                                 transform=True, rand=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        final_lab = []
        final_pre = []
        final_pre_nor = []
        final_name = []
        for batch_idx, (inputs, targets, patient) in enumerate(testloader):
            idx = batch_idx
            #inputs = inputs.permute(0, 4, 1, 2, 3).float()
            inputs = inputs.permute(0, 2, 1, 3, 4).float()

            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            output = net(inputs,3)
            pred = output.data[:,1]


            # _, pred = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(output.data, 1)
            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()


            lab = list(targets.data.cpu().numpy())
            pre = list(pred.data.cpu().numpy())
            pre_nor = list(predicted_com.data.cpu().numpy())
            final_lab.append(lab)
            final_pre.append(pre)
            final_pre_nor.append(pre_nor)
            patient = list(patient)
            final_name.append(patient)
        ff_lab = func(final_lab)
        ff_pre = func(final_pre)
        ff_pre_nor = func(final_pre_nor)
        ff_name = func(final_name)
        final_lab = np.array(ff_lab).flatten()
        final_pre = np.array(ff_pre).flatten()
        file_name = './prediction/%s.mat' % \
                    idx_i[0]
        savemat(file_name, {'name': ff_name, 'lab': final_lab, 'norm': ff_pre_nor, 'pre': final_pre})

