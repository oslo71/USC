import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets
# from torchvision import transforms
import transforms as transforms
from volumentations import *
import pickle
import random
import os
from os import path
from dataloader_USC import lunanod

class transform_TtMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomYFlip(),
            transforms.RandomZFlip()])
        self.strong = Compose([
        # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.2),  # 弹性变形 需注意包括该增强方法在内的一些增强方法对输入数据的维度顺序有要求，不按照指定顺序不知是否会有影响
        GridDropout(ratio=0.5, unit_size_min=2,
                    unit_size_max=4, holes_number_x=4, holes_number_y=4, holes_number_z=2, random_offset=True, p=0.2),

        GaussianNoise(var_limit=(0, 5), p=0.5),
        # # # 让图像中较暗的区域的灰度值得到增强，图像中灰度值过大的区域的灰度值得到降低。经过伽马变换，图像整体的细节表现会得到增强
        # RandomGamma(gamma_limit=(80, 120), p=0.2),

        RandomDropPlane(p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.4),
        Resize((64,64,64), interpolation=1, resize_type=0, p=1.0),
    ], p=1.0)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean), (std))])

    def __call__(self, x):
        weak = self.weak(x)
        data_x = {'image': x}
        strong = self.strong(**data_x)
        return self.normalize(weak), self.normalize(strong['image'])

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
    os.makedirs('dataset/splits', exist_ok=True)
    np.save('dataset/splits/s_n/trainfold.npy', trainfold)
    np.save('dataset/splits/s_n/testfold.npy', testfold)
    return trainfold, testfold

## ('data/datasets', args.n_lbl（有标签数据个数）, lbl_unlbl_split（pkl）, pseudo_lbl_dict（none）（第2轮为pseudo_labeling_iteration_1.pkl）, itr, args.split_txt（'run1'）)
def get_jinshan(args, n_lbl=None, ssl_idx=None, split_txt=''):

    # 第一次取数据后便生成了lbl_unlbl_split（pkl），后面需要在此处加上if判断，如果ssl_idx为none则进入
    ## 此处数据
    k = 5
    dir_jinshan = '/***/imgcutsout/'
    df_jinshan = pd.read_csv(path.join(dir_jinshan, 'jinshan_nooverlap_all.csv'), encoding='utf-8')  # 读取标签数据

    dir_stas = '/***/stas/'
    df_stas = pd.read_csv(path.join(dir_stas, 'annotation-info-merged-realLEN_TT.csv'))
    df_ningbo = pd.read_csv(path.join(dir_stas, 'ningbo_fused_data.csv'))

    id_nodule = []
    id_jinshan_nodule = []
    for i, row in df_jinshan.iterrows():
        id = int(row.Index2cuts)  # 按行读取，读取每一行数据的id编号
        if 'ZS' in row.PatientID or 'P' in row.PatientID:
            id_jinshan_nodule.append(id)

    for i, row in df_stas.iterrows():
        id_stas = int(row.index2cut)
        id_nodule.append(id_stas)
    for i, row in df_ningbo.iterrows():
        id_str = row.index2cut  # 按行读取，读取每一行数据的id编号
        id_nodule.append(id_str)  # id编号不完全等于次序（删除过不全的数据）

    # if not os.path.exists('dataset/splits/splits_stas_ningbo/trainfold.npy'):
    #     random.shuffle(id_nodule)
    #     trafold, tefold = k_fold_split(id_nodule, k)
    # else:
    trafold = np.load('dataset/splits/splits_stas_ningbo/trainfold.npy', allow_pickle=True)
    tefold = np.load('dataset/splits/splits_stas_ningbo/testfold.npy', allow_pickle=True)

    index_fold = int(split_txt[-1])  # 多折交叉，此处未设置循环，手动选取训练和验证集
    id_train = trafold[index_fold]
    id_test = tefold[index_fold]
    # Cal mean std
    preprocesspath = '/***/stas_nb_npy/'
    preprocesspath_jinshan = '/****/local_global/imgcutsout/cutsnpy3D3205/'
    root = preprocesspath

    pixvlu, npix = 0, 0
    countin = 0

    #！！！！！！！！！！！！！！
    ## 结合jinshan和stas，只使用一个mean和std
    # ！！！！！！！！！！！！！！

   ## 对stas数据集进行均值和方差的计算
    for i in id_nodule:
        for fname in os.listdir(preprocesspath):
            if fname.endswith('.npy') and fname[:-4] == str(i):
                data = np.load(os.path.join(preprocesspath, fname))  # 每一个npy文件
                pixvlu += np.sum(data)
                npix += np.prod(data.shape)
                countin += 1
                break
    pixmean = pixvlu / float(npix)  # 对每个像素位置求均值
    print(f'normalization in {countin} training datasets.')
    pixvlu = 0
    for i in id_nodule:
        for fname in os.listdir(preprocesspath):
            if fname.endswith('.npy') and fname[:-4] == str(i):
                data = np.load(os.path.join(preprocesspath, fname)) - pixmean
                pixvlu += np.sum(data * data)
                break
    pixstd = np.sqrt(pixvlu / float(npix))  # 对每个像素位置求方差
    # pixstd /= 255
    print('The mean and stderror of data is:')
    print(pixmean, pixstd)

    id_train = trafold[index_fold][ : int(len(trafold[index_fold])*(n_lbl/100))]
    id_train_another = set(trafold[index_fold])-set(id_train)  # 求完均值方差，再分开

    # Datatransforms
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32 * 2, padding=4),
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
        transforms.Normalize((pixmean), (pixstd)),    ## 使用哪一组标准化数据
    ])

    # transform_jinshan = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((pixmean_jinshan), (pixstd_jinshan)),    ## 使用哪一组标准化数据
    # ])

    trfnamelst = []
    tefnamelst = []
    trlabellst = []
    telabellst = []
    trfnamelst_another = []
    trlabellst_another = []
    trfnamelst_jinshan = []
    trlabellst_jinshan = []

    for i, row in df_stas.iterrows():
        id = int(row.index2cut)
        if id in id_train:
            label = []
            label.append(row.boundary)
            label.append(row.lobulated)
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
            label.append(row.spiculated)
            label.append(row.vessel)
            label.append(row.airbronchogram)
            label.append(row.vpi)
            tefnamelst.append(str(id) + '.npy')  # 读入32*32*32数据
            telabellst.append(label)
        elif id in id_train_another:                   ####   后面再JINSHANSSL类中调用another部分数据时，直接使用jinshan的均值和方差对齐进行了归一化
            # label = [2, 2, 2, 2, 2, 2]                ####   需注意！！！！！！！！！！
            label = []
            label.append(row.boundary)
            label.append(row.lobulated)
            label.append(row.spiculated)
            label.append(row.vessel)
            label.append(row.airbronchogram)
            label.append(row.vpi)
            trfnamelst_another.append(str(id) + '.npy')
            trlabellst_another.append(label)

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
        elif id in id_train_another:                   ####   后面再JINSHANSSL类中调用another部分数据时，直接使用jinshan的均值和方差对齐进行了归一化
            # label = [2, 2, 2, 2, 2, 2]                ####   需注意！！！！！！！！！！
            label = []
            label.append(row.tumor_lung_interface)
            label.append(row.lobulated)
            # label.append(row.cavity | row.vacuole)
            label.append(row.spiculated)
            label.append(row.vessel)
            label.append(row.airbronchogram)
            label.append(row.pi)
            trfnamelst_another.append(str(id) + '.npy')
            trlabellst_another.append(label)

    for i, row in df_jinshan.iterrows():  # 此处选取了所有jinshan数据作为unlbl，上方id_jinshan_nodule只取ZS部分的数据并未对后面的程序造成影响（当时应该是想取其中ZS数据加上zhongshan中的another数据组成unlbl）
        id = int(row.Index2cuts)
        if id in id_jinshan_nodule:
            label = [2,2,2,2,2,2]
            trfnamelst_jinshan.append(str(id) + '.npy')  # 读入32*32*32数据
            trlabellst_jinshan.append(label)

    ## lbl数据中各征象的比例
    print(f'trainset ratio: {np.around(np.sum(trlabellst, axis=0) / len(trlabellst), 3)}')
    print(f'testset ratio: {np.around(np.sum(telabellst, axis=0) / len(trlabellst), 3)}')

    trainset = lunanod(npypath=preprocesspath, fnamelst=trfnamelst, labellst=trlabellst,
                       npypath_jinshan=preprocesspath_jinshan, fnamelst_jinshan=trfnamelst_jinshan, labellst_jinshan=trlabellst_jinshan,
                       fnamelst_another=trfnamelst_another, labellst_another=trlabellst_another,
                       train=True, jinshan=True, another=True, download=True,
                       transform=transform_train)
    testset = lunanod(preprocesspath, tefnamelst, telabellst,
                      npypath_jinshan=None, fnamelst_jinshan=None, labellst_jinshan=None,
                      fnamelst_another=None, labellst_another=None,
                      train=False, jinshan=False, another=False, download=True,
                      transform=transform_test)

    if ssl_idx is None:  # 若还未生成lbl_unlbl_split（pkl）
        # train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(trainset.train_labels, n_lbl)  # 最开始生成过一次，就不会重复再生成了
        train_lbl_idx = set(list(range(0, trainset.train_len)))
        train_unlbl_idx = set(list(range(trainset.train_len, len(trainset.train_labels))))
        train_lbl_idx = [i for i in train_lbl_idx]
        train_lbl_idx = np.array(train_lbl_idx).astype('int64')
        train_unlbl_idx = [i for i in train_unlbl_idx]
        train_unlbl_idx = np.array(train_unlbl_idx).astype('int64')
        # base_dataset = datasets.CIFAR10(root, train=True, download=True)  # 下载初始数据集
        # train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl)  # n_lbl=4000 完成有标签、无标签数据的分割，得到的的对应的索引号

        os.makedirs('dataset/splits', exist_ok=True)
        f = open(os.path.join('dataset/splits', f'stas_ningbo_jinshan_plus_basesplit_{n_lbl}_{split_txt}.pkl'),
                 "wb")  # “wb”：如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}  # lbl_unlbl_dict 中存储内容为有标签、无标签数据的索引
        pickle.dump(lbl_unlbl_dict, f)

    else:  # 若已生成lbl_unlbl_split（pkl）则直接读取
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']  # 有标签数据索引
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']  # 无标签数据索引
        print(f'lbl_data: {len(train_lbl_idx)}')
    ## lbl_data扩增
    train_lbl_idx_initial = train_lbl_idx.copy()
    if len(train_lbl_idx_initial) < args.batch_size*6:  # args.batch_size*6为96，len(train_lbl_idx)为91
        diff = args.batch_size*6 - len(train_lbl_idx_initial)
        train_lbl_idx_initial = np.hstack((train_lbl_idx_initial, np.random.choice(train_lbl_idx_initial,
                                                       diff)))  # np.random.choice用于从ndarray中随机抽取元素，形成指定大小的数组，此处继续通过随即复制，补全正标签样本数量，使其和负标签样本的数量相等
    else:
        assert len(train_lbl_idx_initial) == args.batch_size*6

    if len(train_lbl_idx) < args.batch_size*30:  # args.batch_size*6为96，len(train_lbl_idx)为91
        exapand_labeled = args.batch_size*30 // len(train_lbl_idx)  # 取其整数倍
        train_lbl_idx = np.hstack(
            [train_lbl_idx for _ in range(exapand_labeled)])  # np.hstack将元素按水平方连接，此处即是将lbl_idx复制了exapand_labeled份
        if len(train_lbl_idx) < args.batch_size * 30:
            diff = args.batch_size*30 - len(train_lbl_idx)
            train_lbl_idx = np.hstack((train_lbl_idx, np.random.choice(train_lbl_idx,
                                                       diff)))  # np.random.choice用于从ndarray中随机抽取元素，形成指定大小的数组，此处继续通过随即复制，补全正标签样本数量，使其和负标签样本的数量相等
        else:
            assert len(train_lbl_idx) == args.batch_size*30

    ## unlbl_data扩增
    if len(train_unlbl_idx) < args.batch_size*30:  # args.batch_size*6为480，len(train_lbl_idx)为460
        diff = args.batch_size*30 - len(train_unlbl_idx)
        train_unlbl_idx = np.hstack((train_unlbl_idx, np.random.choice(train_unlbl_idx,
                                                       diff)))  # np.random.choice用于从ndarray中随机抽取元素，形成指定大小的数组，此处继续通过随即复制，补全正标签样本数量，使其和负标签样本的数量相等
    else:
        assert len(train_unlbl_idx) == args.batch_size*30

    lbl_idx_initial = train_lbl_idx_initial.tolist()
    lbl_idx = train_lbl_idx.tolist()

    train_lbl_dataset_initial = JINSHANSSL(  # 返回经过增强的有标签训练数据集，用于dataloader， lbl_idx为有标签数据索引
        root, lbl_idx_initial, train=True, jinshan=True, another=True, transform=transform_train,
        trfnamelst=trfnamelst, trlabellst=trlabellst,
        jinshan_root=preprocesspath_jinshan, trfnamelst_jinshan=trfnamelst_jinshan, trlabellst_jinshan=trlabellst_jinshan,
        trfnamelst_another=trfnamelst_another, trlabellst_another=trlabellst_another)

    train_lbl_dataset = JINSHANSSL(  # 返回经过增强的有标签训练数据集，用于dataloader， lbl_idx为有标签数据索引
        root, lbl_idx, train=True, jinshan=True, another=True, transform=transform_train,
        trfnamelst=trfnamelst, trlabellst=trlabellst,
        jinshan_root=preprocesspath_jinshan, trfnamelst_jinshan=trfnamelst_jinshan, trlabellst_jinshan=trlabellst_jinshan,
        trfnamelst_another=trfnamelst_another, trlabellst_another=trlabellst_another)

    train_unlbl_dataset = JINSHANSSL(  # 返回经过增强的无标签训练数据集，用于dataloader；此处还是读取了其对应标签
        root, train_unlbl_idx, train=True, jinshan=True, another=True, transform=transform_TtMatch((pixmean), (pixstd)),
        trfnamelst=trfnamelst, trlabellst=trlabellst,
        jinshan_root=preprocesspath_jinshan, trfnamelst_jinshan=trfnamelst_jinshan, trlabellst_jinshan=trlabellst_jinshan,
        trfnamelst_another=trfnamelst_another, trlabellst_another=trlabellst_another)

    # test_dataset = datasets.CIFAR10(root, train=False, transform=transform_test, download=True)  # 下载测试集数据10000张，CIFAR10训练集数据50000张
    test_dataset = testset

    return train_lbl_dataset_initial, train_lbl_dataset, train_unlbl_dataset, test_dataset  # 第一次迭代照此返回



def lbl_unlbl_split(lbls, n_lbl):  # (targets, 4000, 10)
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    all_idx = set(list(range(0, len(lbls))))
    idx = random.sample(all_idx, n_lbl)
    np.random.shuffle(idx)
    lbl_idx.extend(idx)
    all_idx -= set(idx)
    unlbl_idx.extend(all_idx)
    return lbl_idx, unlbl_idx


class JINSHANSSL(lunanod):
    def __init__(self, root, indexs, train=True, jinshan=True,another=True,
                 transform=None, target_transform=None,
                 download=True, trfnamelst=None, trlabellst=None,
                 jinshan_root = None, trfnamelst_jinshan=None, trlabellst_jinshan=None, trfnamelst_another=None, trlabellst_another=None):
        super().__init__(root, npypath_jinshan=jinshan_root, train=train, jinshan=jinshan, another=another,# 继承父类init
                         fnamelst=trfnamelst,
                         labellst=trlabellst,
                         fnamelst_jinshan=trfnamelst_jinshan,
                         labellst_jinshan=trlabellst_jinshan,
                         fnamelst_another=trfnamelst_another,
                         labellst_another=trlabellst_another,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.train_labels)

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.train_data[indexs]  # 对应到lunanod类中的def __len__(self)：return len(self.data)
            self.targets = np.array(self.targets)[indexs]  # index不为none时均要读取标签，TT：所使用数据需全部进行标注
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)  # 用于array转为image

        if self.transform is not None :
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target # enumerate(data_loader)时(inputs, targets, indexs, _)即对应这里return的四个值

