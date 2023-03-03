import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import torch


class CenterLoss(torch.nn.Module):
    def __init__(self, class_num, bit):
        super(CenterLoss, self).__init__()
        #10个类中心
        self.hash_targets = self.get_hash_targets(class_num, bit)
        self.criterion = torch.nn.BCELoss()
        self.code_length = bit

    def forward(self, u, y):
        hash_center = self.label2center(y)
        #center_loss = (u - hash_center) ** 2
        center_loss = ((u-hash_center)**2).mean(dim=1)
        return center_loss

    def label2center(self, y):
        #第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸,也就是这里取每行最大值
        #target_temp = torch.Tensor(self.hash_targets).to(y.device)#不能直接赋值，否则会出错，需要中间变量
        hash_center = self.hash_targets[y.argmax(axis=1).cpu().numpy()]
        return torch.from_numpy(hash_center).to(y.device)

    #generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        #H_2K = np.random.permutation(H_2K)
        hash_targets = []
        #只要码平衡的类中心
        for hc in H_2K:
            if np.sum(hc) == 0:
                hash_targets.append(hc)
        #哈达玛类中心不足时，使用伯努利重新采样
        if H_2K.shape[0] < n_class:
            # hash_targets.append(H_K[0])
            # hash_targets.append(-H_K[0])
            #采样N次，如果都没采样到，那就报错吧
            itera = (n_class-H_2K.shape[0]) * 10
            for index in range(itera):
                new_center = self.center_sample(bit)
                #判断新的中心是否在列表中，如果不在,且内积小于bit/2时，则添加进去
                flag = True
                for i in range(len(hash_targets)):
                    if all((hash_targets[i] == new_center)) or hash_targets[i]@new_center>bit/2:
                        flag = False
                        break
                if flag:
                    hash_targets.append(new_center)
                if H_2K.shape[0] == n_class:
                    break
        #如果出现相同的类中心，抛出错误
        hash_targets_test = np.array(list(set([tuple(t) for t in hash_targets])))
        if len(hash_targets_test) != len(hash_targets):
            raise ValueError("The same Center occurred！")

        #拓展思考与优化：1.采样的中心不一定是正交的，2.类中心构建后，能否选出距离最大的N个类中心出来呢？
        #return torch.from_numpy(hash_targets[:n_class]).float()#只要H_2k的前10个为中心
        hash_targets = np.array(hash_targets)
        return hash_targets[:n_class]

    # Bernouli distribution + balance
    def center_sample(self,bit):
        ones = np.ones(bit)
        sa = random.sample(list(range(bit)), bit // 2)
        ones[sa] = -1
        return ones

    def get_center_hash(self):
        return self.hash_targets

    def set_center_hash(self,center_hash):
        self.hash_targets = center_hash

if __name__ == '__main__':
    center_loss = CenterLoss(100,32)

