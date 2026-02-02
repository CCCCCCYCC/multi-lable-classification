import math
import numpy as np
import torch

class AveragePrecisionMeter(object):

    def __init__(self):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        concatenate samples of the new batch and previous batches
        Args:
            output: predicted multiple labels, should be an NxK tensor, postive/negative means presence/absence
            target: ground truth multiple labels, should be an NxK binary tensors, each is multi-hot
        Notes:
            N: the number of samples
            K: the number of classes
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.untyped_storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.untyped_storage().size() * 1.5)
            self.scores.untyped_storage().resize_(int(new_size + output.numel()))
            self.targets.untyped_storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores = self.scores.detach()  # 解除梯度跟踪
        self.targets = self.targets.detach()  # 解除梯度跟踪
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

#    def value(self):
#        """Returns the model's average precision for each class
#        Return:
#            ap (FloatTensor): 1xK tensor, with avg precision for each class k
#        """

#        if self.scores.numel() == 0:
#            return 0
#        self.scores_nonzero = self.scores[:, self.targets.sum(axis=0)a>0]
#        self.targets_nonzero = self.targets[:, self.targets.sum(axis=0)>0]
#        ap = torch.zeros(self.scores_nonzero.size(1))
#        rg = torch.arange(1, self.scores_nonzero.size(0)).float()
#        # compute average precision for each class
#        for k in range(self.scores_nonzero.size(1)):
#            # sort scores
#            scores = self.scores_nonzero[:, k]
#            targets = self.targets_nonzero[:, k]
#            # compute average precision
#            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
#        return ap

    def value(self):
        if self.scores.numel() == 0:
            return torch.zeros(self.num_classes)  # 假设 self.num_classes 是总类别数
        # 计算所有类别的 AP，包括零正样本的类别
        ap = torch.zeros(self.scores.size(1))  # 直接使用总类别数
        for k in range(self.scores.size(1)):
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
        return ap
    @staticmethod
    def average_precision(output, target):
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count==0:
            precision_at_i = 0
        else:
            precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)
    '''''
    def evaluation(self, scores_, targets_):
        """Returns the model's OP, OR, OF1, CP, CR, CF1, EP, ER, EF1
            Return:
            OP, OR, OF1, CP, CR, CF1, EP, ER, EF1: 9 Float tensors
        """
        eps = 1e-10
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))

        OP = np.sum(Nc) / (np.sum(Np) + eps)
        OR = np.sum(Nc) / (np.sum(Ng) + eps)
        OF1 = (2 * OP * OR) / (OP + OR + eps)

        CP = Nc / (Np + eps)
        CR = Nc / (Ng + eps)
        CF1 = (2 * CP * CR) / ( CP + CR + eps)

        CP = np.mean(CP)
        CR = np.mean(CR)
        CF1 = np.mean(CF1)

        # calculate example-based
        pred = np.int8(np.round(1/(1+np.exp(-scores_))))
        gt = np.int8(np.round(targets_))
        TP_e = np.float32(np.sum(((pred+gt) == 2), 1))
        FP_e = np.float32(np.sum(((pred-gt) == 1), 1))
        FN_e = np.float32(np.sum(((pred-gt) == -1), 1))
        TN_e = np.float32(np.sum(((pred+gt) == 0), 1))

        # clear TP_e is 0, assign it some value and latter assign zero
        Nc = TP_e
        Np = TP_e + FP_e
        Ng = TP_e + FN_e

        EP = Nc / (Np + eps)
        ER = Nc / (Ng + eps)
        EF1 = (2 * EP * ER) / (EP + ER + eps)

        EP = np.mean(EP)
        ER = np.mean(ER)
        EF1 = np.mean(EF1)

        return OP, OR, OF1, CP, CR, CF1, EP, ER, EF1
        '''''

    def evaluation(self, scores_, targets_):
        # 固定阈值（0.5）
        pred = (scores_ >= 0.5).astype(int)

        # 全局指标（OP/OR）
        global_TP = np.sum((targets_ == 1) & (pred == 1))
        global_FP = np.sum((targets_ == 0) & (pred == 1))
        global_FN = np.sum((targets_ == 1) & (pred == 0))
        OP = global_TP / (global_TP + global_FP) if (global_TP + global_FP) > 0 else 0
        OR = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0
        OF1 = (2 * OP * OR) / (OP + OR) if (OP + OR) > 0 else 0

        # 类别级指标（CP/CR）
        CP = np.zeros(18)
        CR = np.zeros(18)
        for k in range(18):
            TP_k = np.sum((targets_[:, k] == 1) & (pred[:, k] == 1))
            FP_k = np.sum((targets_[:, k] == 0) & (pred[:, k] == 1))
            FN_k = np.sum((targets_[:, k] == 1) & (pred[:, k] == 0))
            CP[k] = TP_k / (TP_k + FP_k) if (TP_k + FP_k) > 0 else 0
            CR[k] = TP_k / (TP_k + FN_k) if (TP_k + FN_k) > 0 else 0
        CP_avg = np.mean(CP)
        CR_avg = np.mean(CR)
        CF1 = (2 * CP_avg * CR_avg) / (CP_avg + CR_avg) if (CP_avg + CR_avg) > 0 else 0

        # 样本级指标（EP/ER）
        n_samples = targets_.shape[0]
        EP = np.zeros(n_samples)
        ER = np.zeros(n_samples)
        for i in range(n_samples):
            TP_i = np.sum((targets_[i] == 1) & (pred[i] == 1))
            FP_i = np.sum((targets_[i] == 0) & (pred[i] == 1))
            FN_i = np.sum((targets_[i] == 1) & (pred[i] == 0))
            EP[i] = TP_i / (TP_i + FP_i) if (TP_i + FP_i) > 0 else 0
            ER[i] = TP_i / (TP_i + FN_i) if (TP_i + FN_i) > 0 else 0
        EP_avg = np.mean(EP)
        ER_avg = np.mean(ER)
        EF1 = (2 * EP_avg * ER_avg) / (EP_avg + ER_avg) if (EP_avg + ER_avg) > 0 else 0

        # 计算每个类别的 AP 和 mAP
        ap_values = []
        for k in range(18):  # 确保类别数为 18
            scores_k = torch.from_numpy(scores_[:, k])
            targets_k = torch.from_numpy(targets_[:, k])
            ap = self.average_precision(scores_k, targets_k)  # 直接调用类内方法
            ap_values.append(ap)
        mAP = float(np.nanmean(ap_values))

        # 打印调试信息
        print("Class-wise TP/FP:")
        for k in range(18):
            TP_k = np.sum((targets_[:, k] == 1) & (pred[:, k] == 1))
            FP_k = np.sum((targets_[:, k] == 0) & (pred[:, k] == 1))
            print(f"Class {k}: TP={TP_k}, FP={FP_k}, CP={CP[k]:.4f}")

        print("Global TP/FP:", global_TP, global_FP)
        print("OP:", OP)
        print("CP_avg:", CP_avg)
        print("OR:", OR)
        print("CR_avg:", CR_avg)
        print("OF1:", OF1)
        print("CF1:", CF1)
        print("EP_avg:", EP_avg)
        print("ER_avg:", ER_avg)
        print("EF1:", EF1)
        print("mAP:", mAP)

        # 打印每个类别的 AP
        print("Class AP:")
        for idx, ap in enumerate(ap_values):
            print(f"{idx}: {ap:.4f}")

        return OP, OR, OF1, CP_avg, CR_avg, CF1, EP_avg, ER_avg, EF1