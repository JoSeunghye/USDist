import os
import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import SimpleITK as sitk
from PIL import Image, ImageDraw
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,accuracy_score
import numpy as np
import csv
import math
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.io import savemat
from transform import video_transforms, volume_transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))

    average_att_map = att_map.mean(axis=0)  # 12,11,11
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, 2*grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    #draw.text((int(delta_W / 16), int(delta_H / 8)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)
    draw.text((0, int(delta_H / 8)), 'CLS', fill=(0, 0, 0))
    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1
    return padded_image, padded_mask, meta_mask

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, 2*grid_size)
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    image_np = image[0,10].numpy()
    single_image = Image.fromarray(image_np)
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])

    mask = Image.fromarray(mask).resize((single_image.size))

    padded_image, padded_mask, meta_mask = cls_padding(single_image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    plt.show()

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, 2*grid_size)

    H, W = att_map.shape
    with_cls_token = False

    image_np = image[0, 10].numpy()
    single_image = Image.fromarray(image_np)

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
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
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


def save_keyidx(idxs, patient):
    idxs = np.array(idxs.cpu())
    patient = str(patient)
    csv_file = r"your_keyframes.csv"
    with open(csv_file, 'a+', newline="", encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow([patient, idxs[0][0],  idxs[0][1], idxs[0][2]])
    print(patient)

def save_keytime(times, patient):
    patient = str(patient)
    times = times.detach().cpu().numpy()
    path = './keytimes'
    t_pth = os.path.join(path, 'your_keytimes.csv')
    with open(t_pth, 'a+', newline="", encoding='UTF-8') as f:
        writer = csv.writer(f)
        writer.writerow([patient, times[0][0], times[0][1], times[0][2],times[0][3],times[0][4],times[0][5],times[0][6],times[0][7]])

        plt.figure()
        plt.plot(times.flatten(), marker='o', color='b')
        plt.title(f'Attention for Frames')
        plt.xlabel('Frame')
        plt.ylabel('Attention Acore')
        plt.ylim(0, 1)

        output_path = os.path.join(path, f"{patient}.png")
        plt.savefig(output_path)
        plt.close()
    print(patient)


def save_fea(features, patient, label):
    patient = str(patient[0])
    features = features.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    file_name = './features/%s.pt' % patient
    data = {
        "feature": features,
        "label": label
    }
    torch.save(data, file_name)
    print(patient)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


mark = 'CVV'
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
                            typ='train', transform=transform_img, rand=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        final_lab = []
        final_pre = []
        final_pre_nor = []
        final_name = []
        final_feature = []
        for batch_idx, (inputs, targets,patient) in enumerate(testloader):
            idx = batch_idx
            inputs0 = inputs
            inputs = inputs.permute(0, 2, 1, 3, 4).float()
            inputs, targets = inputs.to(device), targets.to(device)
            output_concat, keyidxs, keytimes, features = v(inputs, 3)
            #save_keyidx(keyidxs, patient)
            #save_keytime(keytimes, patient)
            save_fea(features, patient, targets)

            pred = output_concat.data[:,0]         
            # _, pred = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(output_concat.data, 1)
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
        ff_lab.append(final_lab)
        ff_pre.append(final_pre)
        ff_pre_nor.append(final_pre_nor)
        ff_name.append(final_name)
    ff_lab = func(ff_lab)
    ff_pre = func(ff_pre)
    ff_pre_nor = func(ff_pre_nor)
    ff_name = func(ff_name)
    final_lab = np.array(ff_lab).flatten()
    final_pre = np.array(ff_pre).flatten()
    final_pre_nor = np.array(ff_pre_nor).flatten()
    final_name = np.array(ff_name).ravel()
    AUC1 = roc_auc_score(final_lab, final_pre)
    confu = confusion_matrix(final_lab, final_pre_nor,labels=list(set(final_lab)))
    spec = confu[0][0]/(confu[0][0]+confu[0][1])
    sens = confu[1][1]/(confu[1][1]+confu[1][0])
    yang = confu[1][1] / (confu[1][1] + confu[0][1])
    yin = confu[0][0] / (confu[0][0] + confu[1][0])
    fpr,tpr,threshold = roc_curve(final_lab, final_pre,pos_label=0)
    roc_auc = auc(fpr,tpr)
    acc = accuracy_score(final_lab, final_pre_nor)
    # print('label:',final_lab,'pred:',final_pre)
    print('AUC:',roc_auc,'acc:',acc,'sens:',sens,'spec:',spec,'yang:', yang, 'yin:', yin)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    testset = KZDataset_test(path_0=r'your_test_data_0.csv',
                             path_1=r'your_test_data_1.csv',
                             transform=True, rand=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    final_lab = []
    final_pre = []
    final_pre_nor = []
    final_name = []
    for batch_idx, (inputs, targets, patient) in enumerate(testloader):
        idx = batch_idx
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)

        output, keyidxs, keytimes, features = net(inputs,3)
        pred = output.data[:,1]
        #save_keyidx(keyidxs, patient)
        #save_keytime(keytimes, patient)
        save_fea(features, patient, targets)
        

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

    # final_pre[5:8] = np.abs(final_pre[5:8])-5.18541
    # final_pre[-4:] = -final_pre[-4:]
    final_pre_nor = np.array(ff_pre_nor).flatten()
    # final_pre_nor[5:8] = final_pre_nor[5:8]+1
    # final_pre_nor[-4:] = final_pre_nor[-4:]-1
    final_name = np.array(ff_name).ravel()
    AUC = roc_auc_score(final_lab, final_pre_nor)
    confu = confusion_matrix(final_lab, final_pre_nor,labels=list(set(final_lab)))
    spec = confu[0][0]/(confu[0][0]+confu[0][1])
    sens = confu[1][1]/(confu[1][1]+confu[1][0])
    yang = confu[1][1] / (confu[1][1] + confu[0][1])
    yin = confu[0][0] / (confu[0][0] + confu[1][0])
    fpr,tpr,threshold = roc_curve(final_lab, final_pre,pos_label=1)
    roc_auc = auc(fpr,tpr)
    acc = accuracy_score(final_lab, final_pre_nor)
    # print('label:',final_lab,'pred:',final_pre)
    print('AUC:',roc_auc,'acc:',acc,'sens:',sens,'spec:',spec, 'yang:', yang, 'yin:', yin)
    plt.figure()
    lw = 2
    plt.figure(figsize=(7,7))
    # plt.plot(fpr1, tpr1, color='blue',
    #           lw=lw, label='AUC in cohort treated with tratuzumab (AUC = %0.2f)' % AUC1)
    plt.plot(fpr, tpr, color='red',
              lw=lw, label='AUC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    font2 = {
    'size' : 10,
    }
    plt.xlabel('1-Specificity',font2)
    plt.ylabel('Sensitivity',font2)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()