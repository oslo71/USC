'''
USC training:
    在进行USC训练前，首先使用有标签数据训练初始MLC模型（类似main_3D_CNN）；
    接着进行USC training：
         1、载入初始模型；
         2、进行USC training阶段；
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter, accuracy
from utils.utils import *
from datetime import datetime
from USC_dataloading_multi import get_jinshan
from sklearn import metrics

logger = logging.getLogger(__name__)
best_acc = 0

def get_metrics(target, pred):
    # prec, recall, fscore, _ = metrics.precision_recall_fscore_support(target, pred > 0.5,
    #                                                                   average='binary')  # 计算每一类的P,R值
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)  # 绘制ROC曲线，target为真实标签值，pred为预测概率值
    auc = metrics.auc(fpr, tpr)  # 计算AUC值
    return auc

def save_checkpoint_initial(state, is_best, checkpoint, filename='checkpoint_initial.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best_initial.pth.tar'))
        print('successfully saved best currently(initial)-------')

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        print('successfully saved best currently(consistency)-------')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

'''
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 
--expand-labels --seed 5 --out results/cifar10@4000.5
'''

def main(fold_name):
    parser = argparse.ArgumentParser(description='PyTorch USC Training')
    #
    parser.add_argument('--out', default=f'outputs_stas_jinshan_plus_consistency', help='directory to output the result')

    # parser.add_argument('--gpu-id', default='0', type=int,
    #                     help='id(s) for CUDA_VISIBLE_DEVICES')
    # parser.add_argument('--num-workers', type=int, default=4,
    #                     help='number of workers')
    #
    parser.add_argument('--dataset', default='stas_ningbo_jinshan_plus', type=str,  # --dataset "cifar10"
                        help='dataset names')
    #
    parser.add_argument('--n-lbl', type=int, default=25,  # 此处为百分比
                        help='number of labeled data')
    parser.add_argument('--num-labeled', type=int, default=4000,  # CIFAR-10 训练集中每个类别10000张
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    #
    parser.add_argument('--arch', default='cnndense64', type=str,  # --arch "cnn13"
                        choices=['wideresnet', 'cnn13', 'shakeshake', 'cnnres64', 'cnndense64'],
                        help='architecture name')
    parser.add_argument('--initial-epochs', default=100, type=int,  # 1024
                        help='number of initial epochs to run')
    parser.add_argument('--epochs', default=100, type=int,  # 1024
                        help='number of total epochs to run')
    parser.add_argument('--split-txt', default=fold_name, type=str,  # --split-txt "run1"
                        help='use which part of the lbl data')
    parser.add_argument('--cal-c-u-step', default=10, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='train batchsize')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold-c-positive', default=0.75, type=float,
                        help='pseudo label positive threshold')
    parser.add_argument('--threshold-c-negative', default=0.35, type=float,
                        help='pseudo label negative threshold')
    parser.add_argument('--threshold-u-positive', default=0.08, type=float,
                        help='pseudo label positive threshold')
    parser.add_argument('--threshold-u-negative', default=0.12, type=float,
                        help='pseudo label negative threshold')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc

    print(args.split_txt)
    # ## 载入模型
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')  # start time to create unique experiment name
    exp_name = f'exp_{args.dataset}_{args.n_lbl}_{args.arch}_{args.split_txt}_{args.epochs}_{run_started}'
    device = torch.device('cuda')
    args.device = device
    args.exp_name = exp_name
    args.out = os.path.join(args.out, args.exp_name)

    device = torch.device('cuda')
    args.device = device


    os.makedirs(args.out, exist_ok=True)  # 用于存储summarywriter中的信息
    args.writer = SummaryWriter(args.out)

    ## 导入数据集
    if os.path.exists(f'dataset/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'):
        lbl_unlbl_split = f'dataset/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'
    else:
        lbl_unlbl_split = None

    labeled_dataset_initial, labeled_dataset, unlabeled_dataset, test_dataset = get_jinshan(args, args.n_lbl, lbl_unlbl_split, args.split_txt)

    # model = create_model(args)
    # model = nn.DataParallel(model)
    # model.to(args.device)  # 将模型加载到相应的设备中

    # train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader_initial = DataLoader(
        labeled_dataset_initial,
        sampler=RandomSampler(labeled_dataset_initial),
        batch_size=args.batch_size)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size)


    args.start_epoch = 0
    logger.info("***** Running training *****")    # 打印日志
    logger.info(f"  Task = {args.dataset}@{args.n_lbl}")
    logger.info(f"  Num Epochs = {args.initial_epochs}+{args.epochs}")
    # model.zero_grad()
    # test_acc_0, test_mAP_0, test_f1_micro_0, test_f1_macro_0, test_hamming_loss_0, test_auc_0 = train_initial(args, labeled_trainloader_initial ,test_loader, model, optimizer_initial, scheduler_initial)
    # all_results_0 = [test_acc_0, test_mAP_0, test_f1_micro_0, test_f1_macro_0, test_hamming_loss_0]
    # all_auc_0 = test_auc_0
    # resume_path = os.path.join(args.out, 'checkpoint_initial.pth.tar')

    all_results_0 = []
    all_auc_0 = []
    resume_path_3 = '/***/run3_checkpoint_initial.pth.tar'
    resume_path_0 = '/***/run0_checkpoint_initial.pth.tar'
    resume_path_1 = '/***/run1_checkpoint_initial.pth.tar'
    resume_path_4 = '/***/run4_checkpoint_initial.pth.tar'
    resume_path_2 = '/***/run2_checkpoint_initial.pth.tar'
    checkpoint = torch.load(resume_path_1)

    model = create_model(args)

    model.load_state_dict(checkpoint['state_dict'])  # 注意这一句和上一句的位置，不能反，否则key无法匹配
    model = nn.DataParallel(model)
    model.to(args.device)  # 将模型加载到相应的设备中

    print(str(checkpoint['epoch'] - 1) + ': ' + str(checkpoint['acc']) + ' ' + str(checkpoint['best_acc']))
    test_model = model
    test_loss, test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, test_auc = test(args, test_loader,
                                                                                                    test_model)

    # model = nn.DataParallel(model)
    # model.to(args.device)  # 将模型加载到相应的设备中
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model.zero_grad()
    test_acc_1, test_mAP_1, test_f1_micro_1, test_f1_macro_1, test_hamming_loss_1,test_auc_1 = train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, scheduler)
    all_results_1 = [test_acc_1, test_mAP_1, test_f1_micro_1, test_f1_macro_1, test_hamming_loss_1]
    all_auc_1 = test_auc_1
    return all_results_0, all_results_1, all_auc_0, all_auc_1


def train_initial(args, labeled_trainloader, test_loader, model, optimizer, scheduler):

    best_acc_initial = 0
    args.iteration = labeled_trainloader.dataset.train_len // args.batch_size


    for epoch in range(args.start_epoch, args.initial_epochs):
        losses = AverageMeter()
        p_bar = tqdm(range(args.iteration), ncols=120)

        model.train()
        for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
            inputs_x, targets_x = inputs.to(args.device), targets.to(args.device)

            # optimizer.zero_grad()
            targets_x = targets_x.to(args.device)
            logits_x = model(inputs_x)

            lossf = torch.nn.BCELoss()
            loss = lossf(logits_x, targets_x.float())
            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            loss.backward()

            losses.update(loss.item())
            optimizer.step()
            model.zero_grad()

            # mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:4}. Loss: {loss:.4f}.  ".format(
                    epoch=epoch + 1,
                    epochs=args.initial_epochs,
                    batch=batch_idx + 1,
                    iter=args.iteration+1,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    loss=losses.avg,
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        test_model = model

        # if args.local_rank in [-1, 0]:
        test_loss, test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, _ = test(args, test_loader, test_model, epoch)
        scheduler.step()

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        args.writer.add_scalar('test/3.test_mAP', test_mAP, epoch)
        args.writer.add_scalar('test/4.test_f1_micro', test_f1_micro, epoch)
        args.writer.add_scalar('test/5.test_f1_macro', test_f1_macro, epoch)
        args.writer.add_scalar('test/6.hamming_loss', test_hamming_loss, epoch)

        is_best = test_acc > best_acc_initial
        best_acc_initial = max(test_acc, best_acc_initial)

        model_to_save = model.module if hasattr(model, "module") else model

        save_checkpoint_initial({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            # 'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_acc': best_acc_initial,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

    checkpoint = torch.load(f'{args.out}/checkpoint_initial.pth.tar')  # 读取上面保存的模型
    model.load_state_dict(checkpoint['state_dict'], False)
    print(str(checkpoint['epoch'] - 1) + ': ' + str(checkpoint['acc']) + ' ' + str(checkpoint['best_acc']))
    test_model = model
    test_loss, test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, test_auc = test(args, test_loader,
                                                                                              test_model,epoch)
    return test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, test_auc

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, scheduler):

    best_acc_final = 0
    test_accs = []
    end = time.time()

    for epoch in range(args.start_epoch+args.initial_epochs, args.epochs+args.initial_epochs):
        model.train()  # 调整epoch位置（train_TT中还未修改）
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        train_loader = zip(labeled_trainloader, unlabeled_trainloader)
        args.iteration = labeled_trainloader.dataset.data.shape[0] // args.batch_size

        epoch_ct_num = 0
        epoch_p_l_num = 0
        epoch_n_l_num = 0
        epoch_final_tar = []
        epoch_final_mask = []

        p_bar = tqdm(train_loader, ncols=120)
        for batch_idx, (data_x, data_nl) in enumerate(train_loader):

            inputs_x, targets_x = data_x
            # (inputs_u_w, inputs_u_s), _ = data_nl
            (inputs_u_w, inputs_u_s), targets_u = data_nl

            optimizer.zero_grad()
            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]  # 16
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)  # 将有标签数据和w以及s数据尽量均分到每一个batch中
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)  # 还原成torch.cat((inputs_x, inputs_u_w, inputs_u_s))的形式
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)  # 剩下部分截为两半
            # del logits

            ## 进行置信度和不确定性度的判断
            model.eval()
            enable_dropout(model)
            with torch.no_grad():
                out_prob_pl = []
                out_prob_nl = []
                for _ in range(args.cal_c_u_step):  # cal_c_u_step=10
                    outputs_u_w = model(inputs_u_w)
                    out_prob_pl.append(outputs_u_w)
                    out_prob_nl.append(outputs_u_w)
                out_prob_pl = torch.stack(out_prob_pl)  # 堆叠方式合并多个张量
                out_prob_nl = torch.stack(out_prob_nl)
                out_std_pl = torch.std(out_prob_pl, dim=0)  # 每列求标准差
                out_std_nl = torch.std(out_prob_nl, dim=0)
                out_prob_pl = torch.mean(out_prob_pl, dim=0)  # 每列求均值
                out_prob_nl = torch.mean(out_prob_nl, dim=0)

                batch_pl_mask = ((out_std_pl.cpu().numpy() < args.threshold_u_positive) * (out_prob_pl.cpu().numpy() > args.threshold_c_positive)) * 1
                batch_nl_mask = ((out_std_nl.cpu().numpy() < args.threshold_u_negative) * (out_prob_nl.cpu().numpy() < args.threshold_c_negative)) * 1
                batch_pl_mask = np.array(batch_pl_mask)
                batch_nl_mask = np.array(batch_nl_mask)
                batch_pnl_mask = batch_pl_mask | batch_nl_mask
                temp_epoch_ct_num = np.sum(batch_pnl_mask, axis=1)
                epoch_ct_num += temp_epoch_ct_num[temp_epoch_ct_num>0].shape[0]  # 统计存在合格标签的样本数
                batch_pnl_mask = torch.tensor(batch_pnl_mask)

                ## 统计加入计算的伪标签个数
                pl_countin = np.sum(batch_pl_mask)
                nl_countin = np.sum(batch_nl_mask)
                epoch_p_l_num += pl_countin
                epoch_n_l_num += nl_countin

                ## 统计加入计算的伪标签的标签与mask
                temp_tar = targets_u[np.where(batch_pnl_mask == 1)]  # 标志位对应的真实标签
                temp_mask = batch_pl_mask[np.where(batch_pnl_mask == 1)]  # 标签位对应的预测值
                final_posi = np.where(temp_tar != 2)  # 排除标签为2的位置
                final_tar = temp_tar[final_posi]
                final_mask = temp_mask[final_posi]
                epoch_final_tar.extend(final_tar.numpy())
                epoch_final_mask.extend(final_mask)

            model.train()
            ##
            lossfx = torch.nn.BCELoss()
            Lx = lossfx(logits_x, targets_x.float())

            # ##
            # pl_mask = (logits_u_w > args.threshold_c_positive) * 1
            # nl_mask = (logits_u_w < args.threshold_c_negative) * 1
            # pl_mask_copy = np.array(pl_mask.cpu())
            # nl_mask_copy = np.array(nl_mask.cpu())
            # pnl_mask = pl_mask_copy | nl_mask_copy  # 有效位
            # pnl_mask = torch.tensor(pnl_mask)
            #
            # ## 统计加入计算的伪标签个数
            # pl_countin = np.sum(pl_mask_copy)
            # nl_countin = np.sum(nl_mask_copy)
            # epoch_p_l_num += pl_countin
            # epoch_n_l_num += nl_countin
            # ##

            # lossfu = torch.nn.MSELoss()
            lossfu = torch.nn.BCELoss(reduction='none')  # lossfu不求均值
            # Lu = (lossfu(logits_u_s, pl_mask.float()).detach().cpu().numpy() * pnl_mask).mean()
            Lu = (lossfu(logits_u_s, torch.tensor(batch_pl_mask).float().to(args.device)) * batch_pnl_mask.to(args.device)).mean()
            # batch_pnl_mask：伪标签标志位
            # batch_pl_mask：伪标签

            loss = Lx + args.lambda_u * Lu


            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            # scheduler.step()
            # if args.use_ema:
            #     ema_model.update(model)
            # model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            # mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Ir: {lr:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. pseudo_l_num: {p_l: d}/{n_l: d} ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs+args.initial_epochs,
                    batch=batch_idx + 1,
                    iter=args.iteration,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    # data=data_time.avg,
                    # bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    p_l=pl_countin,
                    n_l=nl_countin
                    # mask=mask_probs.avg
                ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()


        test_model = model

        # if args.local_rank in [-1, 0]:
        test_loss, test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, _ = test(args, test_loader, test_model, epoch)
        scheduler.step()

        # 计算整体伪标签正确率
        epoch_final_tar = np.array(epoch_final_tar)
        epoch_final_mask = np.array(epoch_final_mask)
        TP_pseudo = ((epoch_final_tar==1)&(epoch_final_mask==1)).sum()
        TN_pseudo = ((epoch_final_tar==0)&(epoch_final_mask==0)).sum()
        FN_pseudo = ((epoch_final_tar==1)&(epoch_final_mask==0)).sum()
        FP_pseudo = ((epoch_final_tar==0)&(epoch_final_mask==1)).sum()
        acc_all_pseudo = (TP_pseudo+TN_pseudo)/(TP_pseudo+TN_pseudo+FN_pseudo+FP_pseudo)
        acc_posi_pseudo = TP_pseudo/(TP_pseudo+FP_pseudo)  # 值同precision
        acc_nega_pseudo = TN_pseudo/(TN_pseudo+FN_pseudo)

        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        args.writer.add_scalar('train/3.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/4.train_loss_u', losses_u.avg, epoch)
        args.writer.add_scalar('train/5.train_pseudo_ct_num', epoch_ct_num, epoch)
        args.writer.add_scalar('train/6.train_pseudo_pl_num', epoch_p_l_num, epoch)
        args.writer.add_scalar('train/7.train_pseudo_nl_num', epoch_n_l_num, epoch)
        args.writer.add_scalar('train/8.train_pseudo_l_num', epoch_p_l_num+epoch_n_l_num, epoch)
        args.writer.add_scalar('train/9.train_acc_all_pseudo', acc_all_pseudo, epoch)
        args.writer.add_scalar('train/10.train_acc_posi_pseudo', acc_posi_pseudo, epoch)
        args.writer.add_scalar('train/11.train_acc_nega_pseudo', acc_nega_pseudo, epoch)


        args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
        args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
        args.writer.add_scalar('test/3.test_mAP', test_mAP, epoch)
        args.writer.add_scalar('test/4.test_f1_micro', test_f1_micro, epoch)
        args.writer.add_scalar('test/5.test_f1_macro', test_f1_macro, epoch)
        args.writer.add_scalar('test/6.hamming_loss', test_hamming_loss, epoch)



        is_best = test_acc > best_acc_final
        best_acc_final = max(test_acc, best_acc_final)

        model_to_save = model.module if hasattr(model, "module") else model

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            # 'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_acc': best_acc_final,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.out)

    args.writer.close()
    ## 读取最优模型进行测试集结果输出
    checkpoint = torch.load(f'{args.out}/model_best.pth.tar')  # 读取上面保存的模型
    model.load_state_dict(checkpoint['state_dict'], False)
    print(str(checkpoint['epoch']-1)+': '+str(checkpoint['acc'])+' '+str(checkpoint['best_acc']))
    test_model = model
    test_loss, test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, test_auc = test(args, test_loader, test_model,
                                                                                          epoch)
    return test_acc, test_mAP, test_f1_micro, test_f1_macro, test_hamming_loss, test_auc





def test(args, test_loader, model, epoch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    correct = 0
    total = 0
    all_outputs = []  # 在此项目中，验证集即为测试集
    all_targets = []

    if not args.no_progress:
        test_loader = tqdm(test_loader, ncols=120)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            # loss = F.cross_entropy(outputs, targets)
            lossfx = torch.nn.BCELoss()
            loss = lossfx(outputs, targets.float())

            # prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            # top1.update(prec1.item(), inputs.shape[0])
            # top5.update(prec5.item(), inputs.shape[0])

            total += np.array(targets.cpu()).size
            match = (targets.cpu()) == (outputs.cpu() > 0.5)
            correct += torch.sum(match)
            all_outputs.extend(outputs.cpu().squeeze())
            all_targets.extend(targets.cpu().squeeze())

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    # top1=top1.avg,
                    # top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

        all_targets = torch.stack(all_targets)  # 用于将list中的tensor 拼接到一起，变成tensor数组
        all_outputs = torch.stack(all_outputs)
        mAP = metrics.average_precision_score(all_targets.detach().numpy(), all_outputs.detach().numpy())
        f1_macro = metrics.f1_score(all_targets, all_outputs > 0.5, average='macro')
        f1_micro = metrics.f1_score(all_targets, all_outputs > 0.5, average='micro')
        hamming_loss = metrics.hamming_loss(all_targets, all_outputs > 0.5)
        acc = correct.data.item() / total
        print(
            f'Acc in testset is {acc:.3f}, mAP is {mAP:.3f}, f1_mi is {f1_micro:.3f} and f1_ma is {f1_macro:.3f}, hamming_loss is {hamming_loss:.3f}')
        auc_box = []
        for i in range(6):
            auc = get_metrics(all_targets.detach().numpy()[:, i], all_outputs.detach().numpy()[:, i])
            auc_box.append(auc)
            print(f'{auc:.3f}', end=' ')
        print('\n')

    # logger.info("top-1 acc: {:.2f}".format(top1.avg))
    # logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, acc,mAP, f1_micro,f1_macro,hamming_loss, auc_box


if __name__ == '__main__':
    five_fold_name = ['run0', 'run1', 'run2', 'run3', 'run4']
    initial_result, final_result, initial_auc, final_auc = main('run1')


