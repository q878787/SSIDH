import os
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


def calc_similarity_matrix(seen_targets, train_targets):
    S = (train_targets @ seen_targets.t() > 0).float()
    S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
    # Soft similarity matrix, benefit to converge
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def batch_update_B(B, S, U, expand_U, all_B_target,train_targets, args):
    B_label = torch.topk(all_B_target, 1)[1].squeeze(1).cpu().numpy()
    U_label = torch.topk(train_targets, 1)[1].squeeze(1).cpu().numpy()
    start = B_label.min()
    end = B_label.max()
    if end<=10:
        gab = 10
    else:
        gab = 1

    for i in range(start, end, gab):
        B_label_index = np.where((i <= B_label) & (B_label < (i+gab)))[0]
        U_label_index = np.where((i <= U_label) & (U_label < (i+gab)))[0]
        if len(B_label_index)==0 or len(U_label_index)==0:
            print("U_label_index or B_label_index is null")
            continue
        temp_B = B[B_label_index]
        temp_U = U[U_label_index]
        temp_S = S[U_label_index, :][:,B_label_index]

        if expand_U is not None:
            temp_expand_U = expand_U[B_label_index]
            B[B_label_index] = update_B(temp_B, temp_U, temp_S, args.code_length,temp_expand_U, args.alpha)
        else:
            B[B_label_index] = update_B(temp_B, temp_U, temp_S, args.code_length,None, None)
        #print(len(B_label_index),len(U_label_index))
    return B

def update_B(B, U, S, code_length, expand_U=None, gamma=None):
    """
    Solve DCC problem.
    https://zhuanlan.zhihu.com/p/59734411
    """
    # 添加扰动，让他能更好的DCC
    B = ((torch.randn(len(B), code_length).to(B.device))).sign()

    if gamma is None:
        Q = (code_length * S).t() @ U
    else:
        Q = (code_length * S).t() @ U + gamma * expand_U
    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)
        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, targets, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    return code


# 首次使用后，就不需要重新加载了，提高测试速度
global_data = []
global_target = []
global_index = []
def generate_code2(model, dataloader, code_length, class_num, seen_percent, device):
    global global_data, global_target, global_index
    if len(global_data) == 0:
        for data, targets, index in dataloader:
            global_data.append(data)
            global_target.append(targets)
            global_index.append(index)

    model.eval()
    N = len(dataloader.dataset)
    code = torch.zeros([N, code_length])
    query_target = torch.zeros([N, class_num]).double()
    query_index = []
    with torch.no_grad():
        for i in range(len(global_data)):
            data, targets, index = global_data[i], global_target[i], global_index[i]

            data = data.to(device)
            label = torch.topk(targets, 1)[1].squeeze(1)
            temp_list = []
            for j in range(len(index)):
                if label[j] < seen_percent:
                    temp_list.append(int(index[j]))

            if len(temp_list) > 1:
                temp_index = np.array(temp_list) - i * dataloader.batch_size
                hash_code = model(data[temp_index])
                code[index[temp_index], :] = hash_code.sign().cpu()
                query_target[index[temp_index], :] = targets[temp_index]
                query_index.extend(temp_list)

    code = code[query_index]
    target = query_target[query_index]

    return code, target


# 计算 不同hash码的  相似度
def get_similar_probabilities(u, hash_center, bit):
    # get diagonal
    hammin_dist = (bit - torch.diagonal(u @ hash_center.t())) / 2
    probabilities = (bit - hammin_dist) / bit
    return probabilities.view(hash_center.shape[0], 1)


def get_pseudo_label_relaxation_rate(seen_map, it, max_it):
    p = 2
    a = 2 * np.log(99) / (max_it - 1)
    # seen_map越大，sigmoid越往左移，从而提高伪标签数量，或过高的seen_map时可以限制伪标签数量
    c = (1 - seen_map)
    b = c * (np.log(max_it) + a)
    relax = 1 / (1 + np.exp(-a * it + b))
    # 概率越大，反而松弛度越小，所以要1-relax
    if relax < 0.5:
        return 1 - relax / p
    else:
        return 1 - (1 - relax) / p


def get_doesnt_sample_index(all_hit_seen_unmark_omega_index, all_seen_unmark_omega, sample_seen_unmark_omega, batch,batch_size):
    this_batch_omega = get_unified_index_base_omega(all_seen_unmark_omega, torch.IntTensor(
        [i + (batch * int(batch_size)) for i in range(int(batch_size))]).to(all_seen_unmark_omega.device),
                                                    sample_seen_unmark_omega)
    doesnt_sample_index = []
    all_hit_seen_unmark_omega_index_set = set(all_hit_seen_unmark_omega_index)
    for i, omega in enumerate(this_batch_omega):
        if omega not in all_hit_seen_unmark_omega_index_set:
            doesnt_sample_index.append(i)
    return doesnt_sample_index

def get_unmark_label_byNeighbor(hash_center, unmark_u, mark_u, mark_targets, class_num, doesnt_sample_index, device):
    unmark_u = unmark_u.detach().cpu()
    mark_u = mark_u.detach().cpu()
    mark_targets = mark_targets.detach().cpu()

    mark_targets = torch.topk(mark_targets, 1)[1].squeeze(1)  # label
    exist_label = list(set(mark_targets.data.numpy()))  # 获取存在的标签

    labels = np.array([])  # 保存所有标签（不管优不优质）
    labels_prob = np.array([])  # 保存所有标签与类中心的相似度
    good_labels_index = np.array([], dtype=np.int32)  # 保存优质的标签下标
    bad_labels_index = np.array([], dtype=np.int32)  # 保存非优质的标签下标
    hash_center = torch.Tensor(hash_center)
    # 每张图片的特征与 类不同中心输入到 判别器中，获得概率最大的那个为其标签
    for index, u in enumerate(unmark_u):
        repeat_u = u.repeat(class_num, 1)
        probabilitys = get_similar_probabilities(repeat_u, hash_center, unmark_u.shape[1])
        max_probability = probabilitys.squeeze().max().numpy()
        max_prob_label = probabilitys.squeeze().argmax()
        labels = np.append(labels, max_prob_label)
        labels_prob = np.append(labels_prob, max_probability)
        # 如果在已有的标签库中，那么进一步判断
        if max_prob_label.numpy() in exist_label:
            # 如果这个点比所有标注数据的点还要靠近类中心，那么他就是有用的，否则不确定他是否是正确
            marked_u_list = mark_u[mark_targets == max_prob_label]
            csq_center = hash_center[max_prob_label]
            csq_center = csq_center.repeat(marked_u_list.shape[0], 1)
            marked_data_probabilitys = get_similar_probabilities(marked_u_list, csq_center, unmark_u.shape[1])
            # 方式2:比均值还要靠近，那么就选它，但对于ImageNet数据不太可行，因为有100个类别，那1个batch_size才那么几个参考点，甚至没有，不是很准确
            margin = marked_data_probabilitys.mean()
            # margin = margin * pseudo_relax
            if max_probability > margin.numpy() and max_probability > 0.5:
                good_labels_index = np.append(good_labels_index, index)  # 存储最优质的label下标
            else:
                bad_labels_index = np.append(bad_labels_index, index)  # 存储差的label下标
        else:
            bad_labels_index = np.append(bad_labels_index, index)  # 存储最差的label下标
    # margin调控伪标签生成量
    margin = 0.15#1
    if margin==1:
        sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
        good_labels_index = sort_labels_prob_index[len(unmark_u) // 100:]
        bad_labels_index = sort_labels_prob_index[:len(unmark_u) // 100 ]
    else:
        # 如果good或bad其中一个为0，那么分1/4最大/小概率的给对面
        if len(good_labels_index) == 0:
            sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
            good_labels_index = sort_labels_prob_index[len(unmark_u) // 4 * 3:]
            bad_labels_index = sort_labels_prob_index[:len(unmark_u) // 4 * 3]
        elif len(bad_labels_index) == 0:
            sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
            good_labels_index = sort_labels_prob_index[len(unmark_u) // 4:]
            bad_labels_index = sort_labels_prob_index[:len(unmark_u) // 4]
        #限制最低值
        elif len(good_labels_index) / len(unmark_u) < margin:
            # 假设batch为250，good有10个，那么10/250=0.04,sample_num=5000的话，一个epoch为20*10=200个，太少了，所以可以加大margin调控伪标签生成量
            bad_labels_index, good_labels_index = get_good_bad_index(doesnt_sample_index, labels_prob, margin, unmark_u)
            # print(len(good_labels_index)/len(unmark_u),len(unmark_u) * margin)
        #限制最高值
        elif len(good_labels_index) / len(unmark_u) > margin * 2:
            bad_labels_index, good_labels_index = get_good_bad_index(doesnt_sample_index, labels_prob, margin*2, unmark_u)
        #两值之间,且good中存在的doesnt比margin/3还小
        elif len(set(good_labels_index).intersection(set(doesnt_sample_index)))/len(unmark_u) < margin/3:
            middle_margin = len(good_labels_index) / len(unmark_u)
            bad_labels_index, good_labels_index = get_good_bad_index(doesnt_sample_index, labels_prob,middle_margin ,unmark_u)

    # lebel to one-hot
    labels = torch.eye(class_num)[torch.LongTensor(labels), :]
    good_labels_index = torch.from_numpy(good_labels_index)
    bad_labels_index = torch.from_numpy(bad_labels_index)
    return labels.double().to(device), torch.Tensor(labels_prob).to(device), good_labels_index.long().to(
        device), bad_labels_index.long().to(device)

#根据doesnt_sample_index，采样最优的good与较差的bad，其中good有1/3一定是未采样过的数据
def get_good_bad_index(doesnt_sample_index, labels_prob, margin, unmark_u):
    sort_labels_prob_index = np.argsort(labels_prob, )[::-1]  # 从大到小排序
    min_good_count = int(len(unmark_u) * margin)
    good_labels_index, bad_labels_index = [], []
    doesnt_good_index_count = 0
    doesnt_sample_index = set(doesnt_sample_index)
    for spi in sort_labels_prob_index:
        if len(good_labels_index) < min_good_count * 2 / 3:
            good_labels_index.append(spi)
            if spi in doesnt_sample_index:
                doesnt_good_index_count = doesnt_good_index_count + 1
        elif doesnt_good_index_count < min_good_count and spi in doesnt_sample_index:
            good_labels_index.append(spi)
            doesnt_good_index_count = doesnt_good_index_count + 1
        else:
            bad_labels_index.append(spi)
    good_labels_index, bad_labels_index = np.array(good_labels_index), np.array(bad_labels_index)
    return bad_labels_index, good_labels_index


# 通过邻居距离获取符合条件的标签
def get_unmark_label_byNeighbor2(hash_center, unmark_u, mark_u, mark_targets, class_num, device):
    unmark_u = unmark_u.detach().cpu()
    mark_u = mark_u.detach().cpu()
    mark_targets = mark_targets.detach().cpu()

    mark_targets = torch.topk(mark_targets, 1)[1].squeeze(1)  # label
    exist_label = list(set(mark_targets.data.numpy()))  # 获取存在的标签

    labels = np.array([])  # 保存所有标签（不管优不优质）
    labels_prob = np.array([])  # 保存所有标签与类中心的相似度
    good_labels_index = np.array([], dtype=np.int32)  # 保存优质的标签下标
    bad_labels_index = np.array([], dtype=np.int32)  # 保存非优质的标签下标
    hash_center = torch.Tensor(hash_center)
    # 每张图片的特征与 类不同中心输入到 判别器中，获得概率最大的那个为其标签
    for index, u in enumerate(unmark_u):
        repeat_u = u.repeat(class_num, 1)
        probabilitys = get_similar_probabilities(repeat_u, hash_center, unmark_u.shape[1])
        max_probability = probabilitys.squeeze().max().numpy()
        max_prob_label = probabilitys.squeeze().argmax()
        labels = np.append(labels, max_prob_label)
        labels_prob = np.append(labels_prob, max_probability)
        # 如果在已有的标签库中，那么进一步判断
        if max_prob_label.numpy() in exist_label:
            # 如果这个点比所有标注数据的点还要靠近类中心，那么他就是有用的，否则不确定他是否是正确
            marked_u_list = mark_u[mark_targets == max_prob_label]
            csq_center = hash_center[max_prob_label]
            csq_center = csq_center.repeat(marked_u_list.shape[0], 1)
            marked_data_probabilitys = get_similar_probabilities(marked_u_list, csq_center, unmark_u.shape[1])
            # 方式1:比均值还要靠近，那么就选它，但对于ImageNet数据不太可行，因为有100个类别，那1个batch_size才那么几个参考点，甚至没有，不是很准确
            margin = marked_data_probabilitys.mean()
            # margin = margin * pseudo_relax
            if max_probability > margin.numpy() and max_probability > 0.5:
                good_labels_index = np.append(good_labels_index, index)  # 存储最优质的label下标
            else:
                bad_labels_index = np.append(bad_labels_index, index)  # 存储差的label下标
        else:
            bad_labels_index = np.append(bad_labels_index, index)  # 存储最差的label下标
    # margin调控伪标签生成量
    margin = 0.15
    # 如果good或bad其中一个为0，那么分1/4最大/小概率的给对面
    if len(good_labels_index) == 0:
        sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
        good_labels_index = sort_labels_prob_index[len(unmark_u) // 4 * 3:]
        bad_labels_index = sort_labels_prob_index[:len(unmark_u) // 4 * 3]
    elif len(bad_labels_index) == 0:
        sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
        good_labels_index = sort_labels_prob_index[len(unmark_u) // 4:]
        bad_labels_index = sort_labels_prob_index[:len(unmark_u) // 4]
    # 假设batch为250，good有10个，那么10/2500=0.04,sample_num=5000的话，一个epoch为20*10=200个，太少了，所以可以加大margin提高伪标签生成量
    elif len(good_labels_index) / len(unmark_u) < margin:
        sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
        good_labels_index = sort_labels_prob_index[int(len(unmark_u) * (1 - margin)):]
        bad_labels_index = sort_labels_prob_index[:int(len(unmark_u) * (1 - margin))]
        # print(len(good_labels_index)/len(unmark_u),len(unmark_u) * margin)
    #限制伪标签的生成数量
    elif len(good_labels_index) / len(unmark_u) > margin * 2:
        sort_labels_prob_index = np.argsort(labels_prob)  # 从小到大排序
        good_labels_index = sort_labels_prob_index[int(len(unmark_u) * (1 - margin * 2)):]
        bad_labels_index = sort_labels_prob_index[:int(len(unmark_u) * (1 - margin * 2))]

    # lebel to one-hot
    labels = torch.eye(class_num)[torch.LongTensor(labels), :]
    good_labels_index = torch.from_numpy(good_labels_index)
    bad_labels_index = torch.from_numpy(bad_labels_index)
    return labels.double().to(device), torch.Tensor(labels_prob).to(device), good_labels_index.long().to(
        device), bad_labels_index.long().to(device)


# 获取数据的下标，包含可见，不可见，标注和未标注4中
def get_omegas(train_data_len, class_num, num_seen, mark_percent):
    # 生成可见和不可见 数据的序号
    indexs = np.arange(0, train_data_len, 1)
    seen_count = (int)(train_data_len / class_num * num_seen)
    seen_index = indexs[:seen_count]
    unseen_index = indexs[seen_count:]

    # 获得可见和不可见数据的下标  ,从数组中随机排序
    seen_omega = np.random.permutation(seen_index)
    unseen_omega = np.random.permutation(unseen_index)

    # 获得 训练集数据集的 数据下标
    train_seen_mark_omega = seen_omega[:int(len(seen_omega) * mark_percent)]
    train_seen_unmark_omega = seen_omega[int(len(seen_omega) * mark_percent):]
    train_unseen_mark_omega = unseen_omega[:int(len(unseen_omega) * mark_percent)]
    train_unseen_unmark_omega = unseen_omega[int(len(unseen_omega) * mark_percent):]

    return (train_seen_mark_omega, train_seen_unmark_omega, train_unseen_mark_omega, train_unseen_unmark_omega)

#获取all_seen_unmark_omega 的下标
def get_unified_index_base_omega(all_seen_unmark_omega, good_unmark_index, sample_seen_unmark_omega):
    all_seen_unmark_omega = list(all_seen_unmark_omega.cpu().numpy())
    good_unmark_index = good_unmark_index.cpu().numpy()
    # faker_targets_index = []  # ！！！！！！当前 命中的全局下标
    # for om in sample_seen_unmark_omega[good_unmark_index]:
    #     index_temp = np.where(all_seen_unmark_omega == om)[0][0]
    #     faker_targets_index.append(index_temp)
    # 有时会报错
    faker_targets_index = [np.where(all_seen_unmark_omega == om)[0][0] for om in
                           sample_seen_unmark_omega[good_unmark_index]]
    return faker_targets_index


def calc_discrimitor_acc(args, good_unmark_label, good_unmark_targets):
    unmark_targets_temp = torch.topk(good_unmark_targets, 1)[1].squeeze(1)
    unmark_label_temp = torch.topk(good_unmark_label, 1)[1].squeeze(1)
    # 无增量之前，只有num_seen的类别数
    # unmark_targets_temp = unmark_targets_temp[unmark_label_temp < args.num_seen]
    # unmark_label_temp = unmark_label_temp[unmark_label_temp < args.num_seen]
    acc = (unmark_targets_temp == unmark_label_temp).sum().item() / unmark_targets_temp.shape[0]
    return acc


def saveScatterImage(x, y, num, save_path, fileName, class_num=10):
    if class_num > 10:
        class_num = 10

    # path = save_path.split("incre/")[0]+"incre/"
    os.makedirs(save_path, exist_ok=True)
    # 打乱他
    index11 = np.random.permutation(len(x))
    x = x[index11]
    y = y[index11].int()

    x = x[:num].cpu()
    y = torch.topk(y, 1)[1].squeeze(1)[:num].cpu()
    tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=2022)
    X_tsne = tsne.fit_transform(x)
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.06, bottom=0.04, right=0.99, top=0.99)
    for i in range(int(class_num)):
        plt.scatter(X_tsne[y == i][:, 0], X_tsne[y == i][:, 1], s=3, alpha=0.5, label=f'{i}')
        plt.legend()
        plt.savefig(save_path+"/"+fileName + '.eps')
        plt.savefig(save_path + "/" + fileName + '.png')
        plt.savefig(save_path+"/"+fileName + '.pdf')
        plt.savefig(save_path+"/"+fileName + '.svg')
    # plt.show()


def unrepeated_list(all_hit_seen_unmark_omega_index, good_faker_targets_index):
    # 速度快，且顺序不变
    old = set(all_hit_seen_unmark_omega_index)  # 用于查询，set是hash查询，速度快
    for i in good_faker_targets_index:
        if not i in old:
            all_hit_seen_unmark_omega_index.append(i)  # 把不在old的添加进数组里

    return all_hit_seen_unmark_omega_index


# 归一到good_unseen_unmark_labels_prob.min()~1 的范围，即 a~1
def normalization_prob(good_labels_prob):
    a = good_labels_prob.min()
    # 当good_labels_prob仅有1个时，且为1时就会出现问题，加个判断即可
    if int(a) != 1 and len(good_labels_prob) != 1:
        good_labels_prob = a + (good_labels_prob - good_labels_prob.min()) / (
                good_labels_prob.max() - good_labels_prob.min()) * (1 - a)

    return good_labels_prob