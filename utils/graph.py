import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot


tst = np.load('./logs/embeding/1644485702_onlyCenter_incre_binary.npy')
#tst2 = np.load('./logs/embeding/1644462025.6016352_test_incre_binary.npy')
#trn_label = np.loadtxt('output\cifar-10/hard_MyCBAM2_loss2_class_seed2020/trn_label_48.txt') #读取文件a.txt中的数据
tst_label = np.load('./logs/embeding/1644485702_onlyCenter_incre_target.npy')

save_path = "Experimental-Img/graph/DCH0772-cifar"

#随机抽取N个样本
num = 0
if num !=0:
    choice = np.random.randint(0,17700,(num))#54000
    tst = tst[choice]
    tst_label = tst_label[choice]


X = tst
y = tst_label
# X = np.vstack((tst,trn))
# y = np.vstack((tst_label,trn_label))
y = np.argmax(y, axis=1)
# print(y)

'''X是特征，不包含target;X_tsne是已经降维之后的特征'''
tsne = manifold.TSNE(n_components=2, init='pca',n_iter=1000)#random
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}. \
      Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
      
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化


# plt.figure(figsize=(8, 8))

plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.06, bottom=0.04, right=0.99, top=0.99)
for i in range(7,10):
    # print(X_norm[y==i][:,0])
    # print(X_norm[y==i][:,1])
    # print(y==i)
    plt.scatter(X_tsne[y==i][:,0],X_tsne[y==i][:,1],s=3,alpha=0.5,label=f'{i}')

    plt.legend()
    #plt.savefig(save_path + '.eps')
    # plt.savefig(save_path + '.png')
    # plt.savefig(save_path + '.pdf')
    # plt.savefig(save_path + '.svg')


plt.show()




# # 折线图
# hamm
# palette = pyplot.get_cmap('Set1')
# BRE = pd.read_csv('graph/hamm/BRE-CNN.csv')
# DBDH = pd.read_csv('graph/hamm/DBDH.csv')
# DRSCH = pd.read_csv('graph/hamm/DRSCH.csv')
# DSH= pd.read_csv('graph/hamm/DSH.csv')
# DSRH= pd.read_csv('graph/hamm/DSRH.csv')
# HRNH= pd.read_csv('graph/hamm/HRNH.csv')
# KSH = pd.read_csv('graph/hamm/KSH-CNN.csv')
# MLH = pd.read_csv('graph/hamm/MLH.csv')
# RODH = pd.read_csv('graph/hamm/RODH.csv')
# HEGH = pd.read_csv('graph/hamm/HEGH.csv')
# plt.xlim(16, 64) # 限定横轴的范围

# plt.ylim(0, 1) # 限定纵轴的范围
# plt.plot(HEGH['x1'],HEGH['Curve1'], color=palette(0), marker='d', label='RODH')
# plt.plot(RODH['x1'],RODH['Curve1'], color=palette(7), marker='P', label='RODH')
# plt.plot(DBDH['x1'], DBDH['Curve1'], color=palette(2), marker='s', label='DBDH')
# plt.plot(HRNH['x1'],HRNH['Curve1'], color=palette(6), marker='>', label='HRNH')
# plt.plot(DSH['x1'],DSH['Curve1'], color=palette(4), marker='*', label='DSH')
# plt.plot(DRSCH['x1'],DRSCH['Curve1'], color=palette(3), marker='v', label='DRSCH')
# plt.plot(DSRH['x1'],DSRH['Curve1'], color='olive', marker='<', label='DSRH')
# plt.plot(KSH['x1'],KSH['Curve1'],marker='p', color='teal', label='KSH-CNN')
# plt.plot(MLH['x1'],MLH['Curve1'], color=palette(13), marker='x', label='MLH')
# plt.plot(BRE['x1'],BRE['Curve1'], color=palette(1), marker='o', label='BRE-CNN')
# handles, labels = plt.gca().get_legend_handles_labels()
# print(handles)
# print(labels)
# order = [0,2,1]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])








# plt.plot(x, y_Semi, color=palette(2), marker='s', label='Semi-supervised')

# plt.plot(x, y_ours, color=palette(0), marker='o', label='Ours')

# plt.xlabel('#of bits') #X轴标签
#
# plt.ylabel("precision(hamming.dist.<=2)") #Y轴标签
#
# plt.legend(loc='lower left')
#
# pyplot.yticks([0,0.1,0.20, 0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0])
#
# pyplot.xticks([16,24,32,48,64])
#
#
# plt.savefig('hamm.eps',format='eps')

# top
# palette = pyplot.get_cmap('Set1')
# BRE = pd.read_csv('graph/top/BRE-CNN.csv')
# DBDH = pd.read_csv('graph/top/DBDH.csv')
# # DRSCH = pd.read_csv('graph/top/DRSCH.csv')
# DPSH =  pd.read_csv('graph/top/DPSH.csv')
# DSH= pd.read_csv('graph/top/DSH.csv')
# DSRH= pd.read_csv('graph/top/DSRH.csv')
# HRNH= pd.read_csv('graph/top/HRNH.csv')
# KSH = pd.read_csv('graph/top/KSH-CNN.csv')
# MLH = pd.read_csv('graph/top/MLH.csv')
# RODH = pd.read_csv('graph/top/RODH.csv')
# HEGH =  pd.read_csv('graph/top/HEGH.csv')
# plt.xlim(100, 1000) # 限定横轴的范围

# plt.ylim(0, 1) # 限定纵轴的范围
# plt.plot(HEGH ['x1'],HEGH ['Curve1'], color=palette(0), marker='d', label='HEGH ')
# plt.plot(RODH['x1'],RODH['Curve1'], color=palette(7), marker='P', label='RODH')
# plt.plot(DPSH['x1'],DPSH['Curve1'], color=palette(3), marker='v', label='DPSH')
# plt.plot(DBDH['x1'], DBDH['Curve1'], color=palette(2), marker='s', label='DBDH')
# plt.plot(HRNH['x1'],HRNH['Curve1'], color=palette(6), marker='>', label='HRNH')
# plt.plot(DSH['x1'],DSH['Curve1'], color=palette(4), marker='*', label='DSH')

# plt.plot(DSRH['x1'],DSRH['Curve1'], color='olive', marker='<', label='DSRH')
# plt.plot(KSH['x1'],KSH['Curve1'],marker='p', color='teal', label='KSH-CNN')
# plt.plot(MLH['x1'],MLH['Curve1'], color=palette(13), marker='x', label='MLH')
# plt.plot(BRE['x1'],BRE['Curve1'], color=palette(1), marker='o', label='BRE-CNN')
# handles, labels = plt.gca().get_legend_handles_labels()
# print(handles)
# print(labels)

# plt.xlabel('#of retrieved points') #X轴标签
# plt.ylabel('precision') #X轴标签
# # order = [0,2,1]
# # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])








# # plt.plot(x, y_Semi, color=palette(2), marker='s', label='Semi-supervised')

# # plt.plot(x, y_ours, color=palette(0), marker='o', label='Ours')



# pyplot.yticks([0,0.1,0.20, 0.30,0.40,0.50,0.60,0.70,0.80,0.90,1])

# pyplot.xticks([100,200,300,400,500,600,700,800,900,1000])

# plt.legend() # 让图例生效
# plt.savefig('top.eps',format='eps')

