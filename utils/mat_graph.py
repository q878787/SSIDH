import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_mat_graph(mat,file_path):
    plt.matshow(mat, cmap=plt.cm.Reds)#这里设置颜色为红色，也可以设置其他颜色
    plt.colorbar() #颜色标签
    #plt.title("matrix A")

    #不显示刻度，只显示刻度线
    plt.xlabel("Incremental classes")
    plt.ylabel("Origin classes")
    plt.xticks(color='w')
    plt.yticks(color='w')
    #plt.show()

    #plt.savefig(file_path + '.eps')
    # plt.savefig(file_path + '.png')
    # plt.savefig(file_path + '.pdf')
    # plt.savefig(file_path + '.svg')

#左图画SSIDH，右图画DIHN
def create_mat_graph2(ssidh_mat,dihn_mat,file_path,max=5):
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    x= len(dihn_mat)
    #plt.figure(figsize=(8, 4))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,3))#8.5, 3.8
    if len(dihn_mat) % 50 == 0:
        fig.subplots_adjust(left=0.0, bottom=0.08, right=0.98, top=0.84, wspace=0.11, hspace=0.2)
    elif len(dihn_mat) % 30 == 0:
        fig.subplots_adjust(left=0.06, bottom=0.00, right=0.99, top=0.98, wspace=0.15, hspace=0.2)
    elif len(dihn_mat) % 70 == 0:
        fig.subplots_adjust(left=0.06, bottom=0.00, right=0.99, top=0.84, wspace=0.15, hspace=0.2)
        dihn_mat = [list(row) for row in zip(*dihn_mat)]
        ssidh_mat = [list(row) for row in zip(*ssidh_mat)]
    original = x
    incre = 100-x

    #Greens,Blues,terrain,jet    https://wangyeming.github.io/2018/11/07/matplot-cmap/
    color = plt.get_cmap('jet')
    im1 = axes[0].matshow(ssidh_mat, cmap=color,vmin=np.min(ssidh_mat), vmax=max)
    #axes[0].set_title("SSIDH: The number of hits is:"+ str(len([i for i in np.reshape(ssidh_mat,(1,-1))[0] if i<max])))
    axes[0].set_title("SSIDH:原始/增量数据类别数为:"+str(original)+"/"+str(incre),fontsize=16,loc='center',pad="13")
    im2 = axes[1].matshow(dihn_mat, cmap=color,vmin=np.min(dihn_mat), vmax=max)
    axes[1].set_title("DIHN:原始/增量数据类别数为:"+str(original)+"/"+str(incre),fontsize=16,loc='center',pad="13")


    if x% 70 == 0:
        # axes[0].set_ylabel("incremental cluster centers", fontsize=14)
        # axes[0].set_xlabel("original cluster centers", fontsize=14)
        # axes[1].set_ylabel("incremental cluster centers", fontsize=14)
        # axes[1].set_xlabel("original cluster centers", fontsize=14)
        axes[0].set_ylabel("增量数据类中心", fontsize=14)
        axes[0].set_xlabel("原始数据类中心", fontsize=14)
        axes[1].set_ylabel("增量数据类中心", fontsize=14)
        axes[1].set_xlabel("原始数据类中心", fontsize=14)
    else:
        # axes[0].set_xlabel("incremental cluster centers", fontsize=14)
        # axes[0].set_ylabel("original cluster centers", fontsize=14)
        # axes[1].set_xlabel("incremental cluster centers", fontsize=14)
        # axes[1].set_ylabel("original cluster centers", fontsize=14)
        axes[0].set_xlabel("增量数据类中心", fontsize=14)
        axes[0].set_ylabel("原始数据类中心", fontsize=14)
        axes[1].set_xlabel("增量数据类中心", fontsize=14)
        axes[1].set_ylabel("原始数据类中心", fontsize=14)

    if len(dihn_mat) % 50 == 0:
        cbar1 = fig.colorbar(im1, ax=axes[0],shrink=0.5)
        cbar2 = fig.colorbar(im2, ax=axes[1],shrink=0.5)
    else:
        cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.5, orientation='horizontal')
        cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.5, orientation='horizontal')
    cbar1.set_clim(np.min(ssidh_mat), max)
    cbar2.set_clim(np.min(dihn_mat), max)

    plt.savefig(file_path + '.png')
    plt.savefig(file_path + '.pdf')
    plt.savefig(file_path + '.svg')
    plt.show()


#左图画DIHN的矩阵，右图画DIHN的柱状图
def create_mat_graph3(dihn_mat,file_path,max = 5):
    # 设置字体的属性
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    #max = 5 #6.1
    original = len(dihn_mat) % 50
    incre = 100-len(dihn_mat) % 50
    if len(dihn_mat) % 50 == 0:
        plt.figure(figsize=(5, 4.4))#5, 4.4
    elif len(dihn_mat) % 30 == 0:
        plt.figure(figsize=(7, 3))
    elif len(dihn_mat) % 70 == 0:
        plt.figure(figsize=(3.4, 6.7))

    im = plt.matshow(dihn_mat, cmap=plt.get_cmap('jet'),vmin=np.min(dihn_mat), vmax=max, fignum=0)
    hit_num = len([i for i in np.reshape(dihn_mat, (1, -1))[0] if i < max - 1])
    if file_path.find("all")!=-1:
        name = "SIDH"
        hit_num = hit_num-1
    else:
        name = "DIHN"

    # plt.xlabel("incremental cluster centers",fontsize=14)
    # plt.ylabel("original cluster centers",fontsize=14)
    plt.xlabel("增量数据类中心",fontsize=14)
    plt.ylabel("原始数据类中心",fontsize=14)
    # plt.xticks(color='w')
    # plt.yticks(color='w')
    cbar2 = plt.colorbar(im,shrink=0.8)  # 颜色标签
    cbar2.set_clim(np.min(dihn_mat), max)

    if len(dihn_mat)%50==0:
        #plt.title("  "+name + ":K=16,The number of distant<4 is " + str(len([i for i in np.reshape(dihn_mat, (1, -1))[0] if i < max - 1])),fontsize=16)
        plt.title("哈希码长度为16，原始数据与增量数据\n类中心距离小于4的数量有" + str(hit_num)+"个", fontsize=16)
        #plt.title("Distance between cluster centers\n(original classes/incremental classes:" + str(original) + "/" + str(incre) + ")", fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.11, bottom=0.07, right=0.99, top=0.84)#0.95  # 调整子图间距
    elif len(dihn_mat)%30==0:
        #plt.title(name + ":K=16, distant<4:" + str(len([i for i in np.reshape(dihn_mat, (1, -1))[0] if i < max - 1])),fontsize=18)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.10, bottom=0.07, right=0.99, top=0.91)
    elif len(dihn_mat)%70==0:
        #plt.title(name + ":K=16, distant<4:" + str(len([i for i in np.reshape(dihn_mat, (1, -1))[0] if i < max - 1])),fontsize=18)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.16, bottom=0.05, right=0.9, top=0.94)

    plt.savefig(file_path + '.png')
    plt.savefig(file_path + '.pdf')
    plt.savefig(file_path + '.svg')
    plt.show()


#单柱状图
def create_bar_graph(data,file_path):
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(5, 4.4))  # 5, 4.4
    plt.subplots_adjust(wspace=0, hspace=0, left=0.1, bottom=0.11, right=0.995, top=0.9)  # 调整子图间距

    data = np.reshape(data,(1,-1))[0]
    data_set = set(data)#去重
    if file_path.find("all")!=-1:
        data_set = [0]*10 #10
        plt.title("SIDH: Distance between cluster centers\n(original data/incremental data)",fontsize=18)
    else:
        data_set = [0]*5
        #plt.title("DIHN: Distance between cluster centers\n(original data/incremental data)",fontsize=18)
        plt.title("原始数据与增量数据类中心的汉明距离", fontsize=16)
        #plt.title(" "+"\n原始数据与增量数据的类中心距离", fontsize=16)

    #颜色条 https://matplotlib.org/stable/gallery/color/named_colors.html
    for i in range(len(data_set)):
        count = len(np.array(np.where(data==i))[0]) #统计汉明距离为i的有多少个
        plt.bar(i, count, color='tan')
        if count!=0:
            plt.text(i, count + 0.05, '%.0f' % count, ha='center', va='bottom', fontsize=10)

    # plt.xlabel("Hamming Distant" ,fontsize=14)
    # plt.ylabel("Count",fontsize=14)
    plt.xlabel("汉明距离" ,fontsize=14)
    plt.ylabel("数量",fontsize=14)

    plt.savefig(file_path + '.png')
    plt.savefig(file_path + '.pdf')
    plt.savefig(file_path + '.svg')
    plt.show()

#bar柱子上显示数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.14, 1.01*height, '%s' % int(height), size=10, family="Times new roman")

def create_bar_graph2(data,data2,file_path):
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(5, 4))
    #plt.subplots_adjust(wspace=0, hspace=0, left=0.08, bottom=0.11, right=0.99, top=0.86)  # 调整子图间距
    plt.subplots_adjust(wspace=0, hspace=0, left=0.11, bottom=0.12, right=0.92, top=0.86)  # 调整子图间距

    original = len(data)
    incre = len(data[0])

    totalWidth = 0.8  # 一组柱状体的宽度
    labelNums = 2  # 一组有两种类别（例如：DIHN、SIDH）
    barWidth = totalWidth / labelNums  # 单个柱体的宽度
    seriesNums = 6  # 一共有6组（例如：6个汉明距离）

    data = np.reshape(data,(1,-1))[0]
    data2 = np.reshape(data2,(1,-1))[0]
    count = []
    for i in range(seriesNums):
        count.append(len(np.array(np.where(data==i))[0])) #统计汉明距离为i的有多少个
    count2 = []
    for i in range(seriesNums):
        count2.append(len(np.array(np.where(data2==i))[0])) #统计汉明距离为i的有多少个

    #颜色条 https://matplotlib.org/stable/gallery/color/named_colors.html
    cm1=plt.bar([x for x in range(seriesNums)], count, width=barWidth, label="DIHN")
    autolabel(cm1)
    cm2=plt.bar([x + barWidth for x in range(seriesNums)], count2, width=barWidth, label="SIDH")
    autolabel(cm2)

    # plt.xlabel("Hamming Distant" ,fontsize=14)
    # plt.ylabel("Count",fontsize=14)
    plt.xlabel("汉明距离" ,fontsize=14)
    plt.ylabel("数量",fontsize=14)
    #plt.title("Distance between cluster centers\n(original classes/incremental classes:"+str(original)+"/"+str(incre)+")",fontsize=16)
    plt.title("原始数据与增量数据类中心的汉明距离\n(原始/增量数据类别数为:"+str(original)+"/"+str(incre)+")",fontsize=16)
    plt.xticks([x + barWidth / 2 * (labelNums - 1) for x in range(seriesNums)], [str(i) for i in range(seriesNums)])
    plt.legend(loc='upper left', fontsize=16)#显示标签

    plt.savefig(file_path + '.png')
    plt.savefig(file_path + '.pdf')
    plt.savefig(file_path + '.svg')
    plt.show()








def calc_hamming_dist(A,B):
    K = len(A[0])
    return (K-A@B.T)/2


def get_A_and_B(basic_path):
    SEEN_B = torch.load(os.path.join(basic_path, 'seen_b.t')).numpy()
    UNSEEN_B = torch.load(os.path.join(basic_path, 'unseen_b.t')).numpy()
    SEEN_B_TARGETS = torch.load(os.path.join(basic_path, 'seen_targets.t'))
    UNSEEN_B_TARGETS = torch.load(os.path.join(basic_path, 'unseen_targets.t'))

    SEEN_B_TARGETS = torch.topk(SEEN_B_TARGETS, 1)[1].squeeze(1).numpy()
    UNSEEN_B_TARGETS = torch.topk(UNSEEN_B_TARGETS, 1)[1].squeeze(1).numpy()

    label1 = set(SEEN_B_TARGETS)
    label2 = set(UNSEEN_B_TARGETS)

    seen_mean_b = []
    for label in label1:
        index1 = np.array(np.where(SEEN_B_TARGETS == label))[0]
        if np.max(index1)>len(SEEN_B):
            index1 = [i for i in index1 if i<len(SEEN_B)]
        b = SEEN_B[index1]
        meanB = np.sign(np.mean(b, axis=0))
        seen_mean_b.append(meanB)

    unseen_mean_b = []
    for label in label2:
        index1 = np.where(UNSEEN_B_TARGETS == label)
        b = UNSEEN_B[index1]
        meanB = np.sign(np.mean(b, axis=0))
        unseen_mean_b.append(meanB)

    A = np.array(seen_mean_b)
    B = np.array(unseen_mean_b)
    return A,B

def get_center(basic_path,or_classes=0.5):
    center_hash = torch.load(os.path.join(basic_path, 'center_hash.t'))
    origin_center = center_hash[:int(len(center_hash)*or_classes)]
    incre_center = center_hash[int(len(center_hash)*or_classes):]
    return origin_center,incre_center


if __name__ == '__main__':
    basic=50
    basic_path = './c'+str(basic)+'-p1-b16-n5000-bz250-im-all-v1'
    A,B = get_center(basic_path,or_classes=basic/100)
    ssidh_mat = calc_hamming_dist(A, B)

    basic_path = './c'+str(basic)+'-p1-b16-n5000-bz250-im-DIHN-v1'
    A2,B2 = get_A_and_B(basic_path)
    dihn_mat = calc_hamming_dist(A2, B2)

    #双柱状图
    # basic_path = './'+str(basic)+"-bar"
    #create_bar_graph2(dihn_mat,ssidh_mat, basic_path + "-bar")
    #
    # #双矩阵图
    # basic_path = './'+str(basic)+"-mat"
    # ssidh_mat[len(ssidh_mat)-1,len(ssidh_mat[0])-1]=0
    # create_mat_graph2(ssidh_mat, dihn_mat, basic_path,5)

    #柱状图
    #create_bar_graph(dihn_mat, basic_path+"-bar")
    #basic_path = './c' + str(basic) + '-p1-b16-n5000-bz250-im-all-v1'
    #create_bar_graph(ssidh_mat, basic_path+"-bar")

    #矩阵图
    # create_mat_graph3(dihn_mat, basic_path+"-mat",5)
    # basic_path = './c' + str(basic) + '-p1-b16-n5000-bz250-im-all-v1'
    # ssidh_mat[len(ssidh_mat) - 1, len(ssidh_mat[0]) - 1] = 0
    # create_mat_graph3(ssidh_mat, basic_path+"-mat",5)
