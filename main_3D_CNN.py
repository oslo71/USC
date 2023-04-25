'''
3D CNN training
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 多个gpu训练时，每个gpu上的负载时batch_size / n_gpu
import torch
from trainer import Trainer
import torch.optim as optim
from torch.utils.data import DataLoader
import transforms as transforms
from dataloader_multi import lunanod
import os
from datetime import datetime
from models.CNN_DENSE_CBAM import *
import numpy as np
import pandas as pd
from os import path
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from model import generate_model


import random
import ast
import warnings
warnings.filterwarnings("ignore")



def k_fold_split(dataids, k):
    k_fold = []
    trainfold =[]
    testfold = []
    dataids = set(dataids)
    for i in range(k):  # 先将数据集分成k份
        # 防止所有数据不能整除k，最后将剩余的都放到最后一折
        if i == k - 1:
            k_fold.append(list(dataids))
        else:
            tmp = random.sample(list(dataids), int(1.0 / (k-i) * len(dataids)))
            k_fold.append(tmp)
            dataids -= set(tmp)
    # 将原始训练集划分为k个包含训练集和验证集的训练集，同时每个训练集中，训练集：验证集=k-1:1
    for i in range(k):
        # print("第{}折........".format(i + 1))
        tra = []
        testfold.append(k_fold[i])  # 第i部分作测试集
        for j in range(k):
            if i != j:
                tra += k_fold[j]  # 剩余部分作测试集
        trainfold.append(tra)
    print("K-fold done!")
    os.makedirs('data/splits_stas_ningbo', exist_ok=True)             # splits_zhongshan中为stas-zhongshan数据集按照4：1构成训练集和测试集
    np.save('data/splits_stas_ningbo/trainfold.npy', trainfold)
    np.save('data/splits_stas_ningbo/testfold.npy', testfold)
    return trainfold, testfold


#
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
# parser.add_argument('--batch_size', default=2, type=int, help='batch size ')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--savemodel', type=str, default='', help='resume from checkpoint model')
# parser.add_argument("--gpuids", type=str, default='all', help='use which gpu')
#
# parser.add_argument('--num_epochs', type=int, default=100)
# parser.add_argument('--num_epochs_decay', type=int, default=40)
#
# parser.add_argument('--num_workers', type=int, default=16)
#
# parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
# parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
# parser.add_argument('--lamb', type=float, default=1, help="lambda for loss2")
# parser.add_argument('--fold', type=int, default=5, help="fold")
#
# args = parser.parse_args()
#
# resume = False

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# device_ids = range(torch.cuda.device_count())
# print(device_ids)


def model_opt():
    # net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    net = ConvDense([[32, 32], [64, 64]])  # [32, 32], [32, 32], [32, 32]
    # net = generate_model('densenet')
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fK" % (total / 1e3))

    model_dict = net.state_dict()
    checkpoint = torch.load(f'/***/Projects/LUNA16/modelsave/model_3D_luna_malignancy_2.pth.tar')
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict, strict=False)
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)

    # optimizer = optim.Adam(net.parameters(), lr=3e-4)

    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, patience=10)
    print('model building completed------')
    return net, optimizer, sche

if __name__ == '__main__':

    k = 5
    batch_size = 8
    n_epoch = 50

    dir = '/****/dataset/'
    df_stas = pd.read_csv(path.join(dir, 'annotation-info-merged-realLEN_TT.csv'))  # 读取标签数据

    df_ningbo = pd.read_csv(path.join(dir, 'ningbo_fused_data.csv'))


    id_nodule = []
    for i, row in df_stas.iterrows():
        id = int(row.index2cut)  # 按行读取，读取每一行数据的id编号
        id_nodule.append(id)  # id编号不完全等于次序（删除过不全的数据）
    for i, row in df_ningbo.iterrows():
        id_str = row.index2cut  # 按行读取，读取每一行数据的id编号
        id_nodule.append(id_str)  # id编号不完全等于次序（删除过不全的数据）

    # if not os.path.exists('data/splits_stas_ningbo/trainfold.npy'):
    #     random.shuffle(id_nodule)
    #     trafold, tefold = k_fold_split(id_nodule, k)
    # trafold = np.load('data/splits_stas_ningbo/trainfold.npy', allow_pickle=True)
    # tefold = np.load('data/splits_stas_ningbo/testfold.npy', allow_pickle=True)

    # if not os.path.exists('data/splits/splits_stas_ningbo/trainfold.npy'):
    #     random.shuffle(id_nodule)
    #     trafold, tefold = k_fold_split(id_nodule, k)
    # else:
    #     trafold = np.load('data/splits/splits_stas_ningbo/trainfold.npy', allow_pickle=True)
    #     tefold = np.load('data/splits/splits_stas_ningbo/testfold.npy', allow_pickle=True)
    trafold = np.load('data/splits_stas_ningbo/trainfold.npy', allow_pickle=True)
    tefold = np.load('data/splits_stas_ningbo/testfold.npy', allow_pickle=True)

    max_mAP = []
    max_mAP_epoch = []
    X_mAP_f1_mi = []
    X_mAP_f1_ma = []

    end_acc = []
    end_mAP = []
    end_f1_mi = []
    end_f1_ma = []
    end_hamming_loss = []
    end_auc = []

    summarytime = datetime.today().strftime('%d-%m-%y_%H%M')

    for index_fold in range(k):
        # if index_fold != 2:
        #     continue
        id_train = trafold[index_fold]
        id_test = tefold[index_fold]

        # Cal mean std
        preprocesspath_stas_nb = f'/***/dataset/stas_nb_npy/'
        pixvlu, npix = 0, 0
        countin = 0
        for i in id_nodule:  # 需要修改，应只在训练集数据上做标准化，测试集使用训练集数据进行标准化
            for fname in os.listdir(preprocesspath_stas_nb):
                if fname.endswith('.npy') and fname[:-4] == str(i):
                    data = np.load(os.path.join(preprocesspath_stas_nb, fname))  # 每一个npy文件
                    pixvlu += np.sum(data)
                    npix += np.prod(data.shape)
                    countin += 1
                    break
        pixmean = pixvlu / float(npix)  # 对每个像素位置求均值
        print(f'normalization in {countin} training datasets.')
        pixvlu = 0
        for i in id_nodule:
            for fname in os.listdir(preprocesspath_stas_nb):
                if fname.endswith('.npy') and fname[:-4] == str(i):
                    data = np.load(os.path.join(preprocesspath_stas_nb, fname)) - pixmean
                    pixvlu += np.sum(data * data)
                    break
        pixstd = np.sqrt(pixvlu / float(npix))  # 对每个像素位置求方差
        # pixstd /= 255
        print('The mean and stderror of data is:')
        print(pixmean, pixstd)
        # logging.info('mean ' + str(pixmean) + ' std ' + str(pixstd))

        # Datatransforms
        # logging.info('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
        transform_train = transforms.Compose([
            # transforms.RandomScale(range(28, 38)),
            # transforms.RandomCrop(32 * 2, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomYFlip(),
            transforms.RandomZFlip(),
            # transforms.ZeroOut(4),
            transforms.ToTensor(),
            transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
        ])
        ## 此处验证集也使用训练集的均值和方差进行标准化
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((pixmean), (pixstd)),
        ])


        savemodelpath = '/***/Projects/USC/modelsave_multi_review_50/fold-' + str(index_fold) + '/'
        if not os.path.exists(savemodelpath):
            os.makedirs(savemodelpath)

        print(f'The {index_fold+1} fold experiment------ ')
        # load data list
        # namelst = []
        # labellst = []
        trfnamelst = []
        tefnamelst = []
        trlabellst = []
        telabellst = []

        ## 读入stas征象标签数据（此处这样写会导致train和test中的顺序又变为有序，不过后面dataloader中会打乱，整体不影响）
        for i, row in df_stas.iterrows():
            id = int(row.index2cut)
            if id in id_train:
                label = []
                label.append(row.boundary)
                label.append(row.lobulated)
                # label.append(row.cavity | row.vacuole)
                label.append(row.spiculated)
                label.append(row.vessel)
                label.append(row.airbronchogram)
                label.append(row.vpi)
                trfnamelst.append(str(id) + '.npy')  # 读入32*32*32数据
                trlabellst.append(label)
            elif id in id_test:
                label = []
                label.append(row.boundary)
                label.append(row.lobulated)
                # label.append(row.cavity | row.vacuole)
                label.append(row.spiculated)
                label.append(row.vessel)
                label.append(row.airbronchogram)
                label.append(row.vpi)
                tefnamelst.append(str(id) + '.npy')  # 读入32*32*32数据
                telabellst.append(label)

        ## 读入ningbo征象标签数据
        for i, row in df_ningbo.iterrows():
            id = row.index2cut
            if id in id_train:
                label = []
                label.append(row.tumor_lung_interface)
                label.append(row.lobulated)
                # label.append(row.cavity | row.vacuole)
                label.append(row.spiculated)
                label.append(row.vessel)
                label.append(row.airbronchogram)
                label.append(row.pi)
                trfnamelst.append(str(id) + '.npy')  # 读入32*32*32数据
                trlabellst.append(label)
            elif id in id_test:
                label = []
                label.append(row.tumor_lung_interface)
                label.append(row.lobulated)
                # label.append(row.cavity | row.vacuole)
                label.append(row.spiculated)
                label.append(row.vessel)
                label.append(row.airbronchogram)
                label.append(row.pi)
                tefnamelst.append(str(id) + '.npy')  # 读入32*32*32数据
                telabellst.append(label)

        # from sklearn.model_selection import train_test_split
        # trfnamelst, tefnamelst, trlabellst, telabellst = train_test_split(namelst, labellst, test_size=0.2,shuffle=True,random_state=1)

        trainset = lunanod(preprocesspath_stas_nb, trfnamelst, trlabellst, train=True, val=False, download=True,
                           transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = lunanod(preprocesspath_stas_nb, tefnamelst, telabellst, train=False, val=False, download=True,
                          transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        print("Data loader completed-----")
        print(f'trainset ratio: {np.around(np.sum(trainset.train_labels, axis=0) / trainset.train_labels.shape[0], 3)}')
        print(f'testset ratio: {np.around(np.sum(testset.test_labels, axis=0) / testset.test_labels.shape[0], 3)}')

        model, opt, sche = model_opt()
        trainer3D = Trainer(trainloader, testloader, batch_size=batch_size, n_epochs=n_epoch,
                            net=model, optimizer=opt, scheduler=sche, loss=nn.BCELoss(),
                            device=torch.device('cuda'), name='3D experiments', trainsize=len(trainset), testsize=len(testset),
                            savemodelpath=savemodelpath, deterministic=True, parallel=True)
        Acc, mAP, f1_ma, f1_mi, hamming_loss, auc = trainer3D.run(index_fold, summarytime)  # MLC
        # Acc, auc = trainer3D.run(index_fold, summarytime)
        end_acc.append(Acc)
        end_mAP.append(mAP)
        end_f1_ma.append(f1_ma)
        end_f1_mi.append(f1_mi)
        end_hamming_loss.append(hamming_loss)
        end_auc.append(auc)

    ## MLC
    print(f'Acc: {np.mean(end_acc)}; mAP: {np.mean(end_mAP)}; f1_ma: {np.mean(end_f1_ma)}; f1_mi: {np.mean(end_f1_mi)}; hamming_loss: {np.mean(end_hamming_loss)}')
    # print(
    #     f'Acc: {np.mean(end_acc)}')

    print(f'the average AUC: {np.mean(end_auc, axis=0)}')



