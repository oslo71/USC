# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

import time
from sklearn import metrics
import os
from torch.utils.tensorboard import SummaryWriter
import shutil

global best_acc

sign_class = 6

class Trainer:
    def __init__(self, training_set,
                 validation_set,
                 batch_size,
                 n_epochs,
                 net,
                 optimizer,
                 scheduler,
                 loss,
                 name,
                 trainsize,
                 testsize,
                 savemodelpath,
                 device,  # cuda
                 index_fold=None,
                 deterministic=False,
                 parallel=True,  # False
                 ifinference = False,

                 ):

        torch.backends.cudnn.deterministic = deterministic
        self.batch_size = batch_size
        self.train_dataset = training_set
        self.valid_dataset = validation_set
        self.test_dataset = validation_set
        self.net = net
        self.device = device
        # self.net.to(self.device)
        self.parallel = parallel
        if parallel:
            self.net = nn.DataParallel(net)
        self.net.to(self.device)
        if ifinference:
            checkpoint = torch.load(os.path.join('/***/Projects/USC/modelsave_multi_densecbam',
                                                 'fold-' + str(index_fold) + '/checkpoint.pth.tar'))
            loaded_dict = checkpoint['state_dict']
            self.net.load_state_dict(loaded_dict)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.n_epochs = n_epochs
        self.name = name
        self.trainsize = trainsize
        self.valsize = testsize
        self.testsize = testsize
        self.savemodelpath = savemodelpath
        self.log = ''

    def save_checkpoint(self, state, is_best, checkpoint, filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint,
                                                   'model_best.pth.tar'))
            print('successfully saved best currently-------')

    def get_metrics(self, target, pred):
        prec, recall, fscore, _ = metrics.precision_recall_fscore_support(target, pred > 0.5,
                                                                          average='binary')  # 计算每一类的P,R值
        fpr, tpr, thresholds = metrics.roc_curve(target, pred)  # 绘制ROC曲线，target为真实标签值，pred为预测概率值
        auc = metrics.auc(fpr, tpr)  # 计算AUC值
        return prec, recall, auc, fscore

    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        max_mAP = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_dataset):
            inputs, targets = inputs.to(self.device), targets.to(self.device)  # 原来此处是inputs.cuda()，不能实现在指定的GPU上进行训练，总会将数据放在GPU0上
            # targets = targets.long()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            targets = torch.squeeze(targets)
            # outputs, hidden = net(inputs)
            outputs = self.net(inputs)
            outputs = torch.squeeze(outputs)
            loss = self.loss(outputs, targets.float())

            loss.backward()
            self.optimizer.step()
            train_loss += loss.data.item()
            # _, predicted = torch.max(outputs.data, 1)  # dim = 1 输出所在行的最大值
            total += np.array(targets.cpu()).size
            match = (targets.cpu()) == (outputs.cpu() > 0.5)
            correct += torch.sum(match)

        print('epoch ' + str(epoch))
        print('tracc ' + str(correct.data.item() / float(total)))
        val_loss, acc_v, f1_ma_v, f1_mi_v, mAP_v = self.test(epoch, self.savemodelpath)  # 测试集作验证集

        # 返回每个epoch的准确率 = 正确预测的数量/总样本数
        return train_loss/(batch_idx + 1), val_loss, acc_v, f1_ma_v, f1_mi_v, mAP_v


    def test(self, epoch,savemodelpath):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_outputs = torch.zeros([self.valsize, sign_class])  # 在此项目中，验证集即为测试集
        all_targets = torch.zeros([self.valsize, sign_class])
        for batch_idx, (inputs, targets) in enumerate(self.valid_dataset):
            with torch.no_grad():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
                outputs = self.net(inputs)
                loss = self.loss(outputs, targets.float())
            test_loss += loss.data.item()
            total += np.array(targets.cpu()).size
            match = (targets.cpu()) == (outputs.cpu() > 0.5)
            correct += torch.sum(match)
            st = batch_idx * self.batch_size
            all_targets[st:st + outputs.shape[0], :] = targets.cpu().squeeze()
            all_outputs[st:st + outputs.shape[0], :] = outputs.cpu().squeeze()
        mAP = metrics.average_precision_score(all_targets.detach().numpy(), all_outputs.detach().numpy())
        f1_macro = metrics.f1_score(all_targets, all_outputs > 0.5, average='macro')
        f1_micro = metrics.f1_score(all_targets, all_outputs > 0.5, average='micro')
        acc = correct.data.item() / total

        #  predict 和 label 同时为1
        TP = ((all_outputs > 0.5) & (all_targets == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN = ((all_outputs < 0.5) & (all_targets == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN = ((all_outputs < 0.5) & (all_targets == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP = ((all_outputs > 0.5) & (all_targets == 0)).cpu().sum()

        sen = TP / (TP + FN)
        spe = TN/(FP+TN)

        print('The validation set:')
        print('teacc ' + str(acc) + ' mAP ' + str(mAP))
        print('f1-macro: ' + str(f1_macro))
        print('f1-micro: ' + str(f1_micro))
        print(f'sensitivity: {sen}; specificity: {spe}')
        print("The corresponding AUC of each CT sign is as follows:")
        for i in range(sign_class):
            prec, recall, auc, fscore = self.get_metrics(all_targets.detach().numpy()[:, i],
                                                    all_outputs.detach().numpy()[:, i])  # 得到模型在测试集上的P，R，AUC值
            print(f'{auc:.3f}', end=' ')
        return test_loss/(batch_idx + 1), acc, f1_macro, f1_micro, mAP

    def predict_test(self):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_outputs = torch.zeros([self.testsize, sign_class])  # 在此项目中，验证集即为测试集
        all_targets = torch.zeros([self.testsize, sign_class])
        for batch_idx, (inputs, targets) in enumerate(self.test_dataset):
            with torch.no_grad():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
                outputs = self.net(inputs)
                loss = self.loss(outputs, targets.float())
            test_loss += loss.data.item()
            total += np.array(targets.cpu()).size
            match = (targets.cpu()) == (outputs.cpu() > 0.5)
            correct += torch.sum(match)
            st = batch_idx * self.batch_size
            all_targets[st:st + outputs.shape[0], :] = targets.cpu().squeeze()
            all_outputs[st:st + outputs.shape[0], :] = outputs.cpu().squeeze()
        mAP = metrics.average_precision_score(all_targets.detach().numpy(), all_outputs.detach().numpy())
        f1_macro = metrics.f1_score(all_targets, all_outputs > 0.5, average='macro')
        f1_micro = metrics.f1_score(all_targets, all_outputs > 0.5, average='micro')
        hamming_loss = metrics.hamming_loss(all_targets, all_outputs > 0.5)
        acc = correct.data.item() / total

        #  predict 和 label 同时为1
        TP = ((all_outputs > 0.5) & (all_targets == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN = ((all_outputs < 0.5) & (all_targets == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN = ((all_outputs < 0.5) & (all_targets == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP = ((all_outputs > 0.5) & (all_targets == 0)).cpu().sum()

        sen = TP / (TP + FN)
        spe = TN / (FP + TN)

        print('The testing set:')
        print('teacc ' + str(acc) + ' mAP ' + str(mAP))
        print('f1-macro: ' + str(f1_macro))
        print('f1-micro: ' + str(f1_micro))
        print(f'sensitivity: {sen}; specificity: {spe}')
        print("The corresponding AUC of each CT sign is as follows:")
        auc_box = []
        for i in range(sign_class):
            prec, recall, auc, fscore = self.get_metrics(all_targets.detach().numpy()[:, i],
                                                         all_outputs.detach().numpy()[:, i])  # 得到模型在测试集上的P，R，AUC值
            auc_box.append(auc)
            print(f'{auc:.3f}', end=' ')
        return test_loss / (batch_idx + 1), acc, f1_macro, f1_micro, mAP, hamming_loss, auc_box, all_targets, all_outputs




    def run(self, index, summarytime):
        start_t = time.time()
        summarypath = os.path.join('/****/Projects/USC/lossresults_multi_densecbam/',summarytime)
        os.makedirs(summarypath, exist_ok=True)
        writer = SummaryWriter(summarypath)
        max_f1_ma = 0
        max_f1_mi = 0
        max_mAP = 0
        best_acc_final = 0  # 用于保存测试集上最好结果
        for epoch in range(self.n_epochs):
            trainloss, valloss,acc, f1_ma, f1_mi, mAP= self.train(epoch)  # 训练过程中epoch x：的输出代码位于该函数中的report函数
            print(f'loss on training set:{trainloss}, on valid set:{valloss}')
            print('\n')
            self.scheduler.step(valloss)
            # writer.add_scalar(f'pic_{index}', trainloss, global_step=epoch)
            writer.add_scalars(f'pic_{index}', {'trainloss': trainloss, 'valloss': valloss}, global_step=epoch)

            is_best = acc > best_acc_final
            best_acc_final = max(acc, best_acc_final)
            model_to_save = self.net
            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'acc': acc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, is_best, self.savemodelpath)

        testloss,acc_t, f1_macro_t, f1_micro_t, mAP_t, hamming_loss_t, auc_t, _, _= self.predict_test()


        diff = time.time() - start_t
        print(f'took {diff} seconds')
        # print(f'max_f1_ma: {max_f1_ma} in {max_f1_ma_epoch}; max_f1_mi: {max_f1_mi} in {max_f1_mi_epoch}; max_mAP: {max_mAP} in {max_mAP_epoch}')
        return acc_t, mAP_t, f1_macro_t, f1_micro_t, hamming_loss_t, auc_t

    def inference(self):
        testloss,acc_t, f1_macro_t, f1_micro_t, mAP_t, hamming_loss_t, auc_t, all_targets_t, all_outputs_t = self.predict_test()
        # print(f'max_f1_ma: {max_f1_ma} in {max_f1_ma_epoch}; max_f1_mi: {max_f1_mi} in {max_f1_mi_epoch}; max_mAP: {max_mAP} in {max_mAP_epoch}')
        return acc_t, mAP_t, f1_macro_t, f1_micro_t, hamming_loss_t, auc_t,all_targets_t, all_outputs_t



