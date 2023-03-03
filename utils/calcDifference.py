import torch
import os
import numpy as np


# SEEN_MARK_B1 = torch.load(os.path.join('checkpoints', 'seen_mark_b.t'))
# SEEN_UNMARK_B1 = torch.load(os.path.join('checkpoints', 'seen_unmark_b.t'))
# SEEN_MARK_B2 = torch.load(os.path.join('iteratorData', args.now, "incre", "final", 'seen_mark_b.t'))
# SEEN_UNMARK_B2 = torch.load(os.path.join('iteratorData', args.now, "incre", "final", 'seen_unmark_b.t'))
# SEEN_B1 = torch.cat((SEEN_MARK_B1,SEEN_UNMARK_B1),dim=0)
# SEEN_B2 = torch.cat((SEEN_MARK_B2,SEEN_UNMARK_B2),dim=0)

SEEN_B1 = torch.load(os.path.join('../checkpoints', 'temp', 'old_B0.t'))
SEEN_B2 = torch.load(os.path.join('../checkpoints', 'temp', 'old_B_incre0.t'))
SEEN_TARGETS = torch.load(os.path.join('../checkpoints', 'temp', 'old_targets0.t'))
#onehot转label
labels = torch.topk(SEEN_TARGETS, 1)[1].squeeze(1)

b1 = SEEN_B1.numpy()
b2 = SEEN_B2.numpy()
targets = labels.numpy()

#每一个图片的hash码变化率
rate = 0
for i in range(len(b1)):
    count = 0
    for j in range(len(b1[i])):
        if b1[i][j] != b2[i][j]:
            count= count+1
    #不相同个数的比率
    diff_rate = count/len(b1[i])
    rate = rate+diff_rate
rate = rate/len(b1)
print(f"每张图像hash码平均变化率：%f.3",rate)

#每一个类别hash码变化率
class_num = 7
#每一个类中心的hash变化率
class_diff_rate = []
#类中心移动的汉明距离
class_move_dist = []
for i in range(class_num):
    index = np.where(i==targets)
    b11 = b1[index]
    b22 = b2[index]
    b11 = b11.mean(axis=0)
    b22 = b22.mean(axis=0)
    #hash码的变
    count = 0
    for i in range(len(b11)):
        if b11[i]!=b22[i]:
            count = count+1
    label_diff_rate = count/len(b11)
    class_diff_rate.append(label_diff_rate)
    class_move_dist.append(int(label_diff_rate*len(b11)))

print("每个类别hash center的变化率：%f.3",class_diff_rate)
print("每个类别hash center移动的汉明距离：%f",class_move_dist)





