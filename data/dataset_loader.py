import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from data.transform import train_transform
from data.cifar10 import load_cifar10_dataloader
from data.imagenet100 import load_imagenet100_dataloader
from concurrent.futures import ThreadPoolExecutor, as_completed

#获取数据集的dataloader
def load_data(dataset,root,batch_size,num_workers,omegas):
    if dataset == 'cifar-10':
        query_dataloader, seen_mark_dataloader, seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader, retrieval_dataloader =\
            load_cifar10_dataloader(root,batch_size,num_workers,omegas)
    elif dataset == 'imagenet-100':
        query_dataloader, seen_mark_dataloader, seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader, retrieval_dataloader =\
            load_imagenet100_dataloader(root,batch_size,num_workers,omegas)
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, seen_mark_dataloader, seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader, retrieval_dataloader


#随机从训练集中采样 部分数据来训练
def sample_dataloader(dataloader, num_samples, batch_size, root, dataset,has_sampled_index=None,sampled_percent=0.2):
    '''
    :param dataloader:
    :param num_samples:
    :param batch_size:
    :param root:
    :param dataset:
    :param has_sampled_index:  已经采样过的全局omega下标
    :param sampled_percent:
    :return:
    '''
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets
    dlomega = dataloader.dataset.omega

    #随机采样，原则是，1.已经采样过的部分采样，2.未采样过的也部分采样
    has_sampled_index_list = [] #存取与random_omega同类型的下标
    doesnot_sampled_index_list = []
    #如果首次采样，那么直接全部填充进未采样过的数组中去
    if has_sampled_index is None or len(has_sampled_index)==0:
        random_omega = np.random.permutation(len(data))
        doesnot_sampled_index_list = random_omega.tolist()[:num_samples]
    else: #否则先从已采样数组中采样一半的数据，再从未采样数组中采样一半数据
        random_omega = np.random.permutation(len(data))
        rest_sample_num = num_samples #剩余需采样的数据
        #从已采样过的数据中 重新采样一部分数据
        temp_has_sampled_index_list = np.random.permutation(has_sampled_index).tolist()
        if int(num_samples*sampled_percent)<=len(has_sampled_index):
            #采样sampled_percent比例数据
            has_sampled_index_list = temp_has_sampled_index_list[:int(num_samples*sampled_percent)]
            temp_has_sampled_index_list = temp_has_sampled_index_list[int(num_samples*sampled_percent):] #截断
            rest_sample_num = rest_sample_num-int(num_samples*sampled_percent)
        else:
            has_sampled_index_list = temp_has_sampled_index_list[:len(has_sampled_index)]
            temp_has_sampled_index_list = temp_has_sampled_index_list[len(has_sampled_index):]
            rest_sample_num = rest_sample_num - len(has_sampled_index)

        #从未采样过的数据中 采样剩余所需数据  (这里使用交集的方式，循环太慢了)
        doesnot_sampled_index_list = list(set(random_omega).difference(set(has_sampled_index)))[:rest_sample_num]
        rest_sample_num = rest_sample_num-len(doesnot_sampled_index_list)
        # 如果从未标注数据采样完还是不够sample_num，那么就从已经采样过的数据中填充
        if rest_sample_num > 0:
            has_sampled_index_list.extend(temp_has_sampled_index_list[:rest_sample_num])

    #采样和未采样的数据整合在一起
    has_sampled_index_list.extend(doesnot_sampled_index_list)

    # 随机从抽样长度中 再次打乱抽取 下标
    samples_omega_index = np.random.permutation(has_sampled_index_list)

    data = data[samples_omega_index]
    targets = targets[samples_omega_index]
    omega = dlomega[samples_omega_index]  # 从这堆元素中获得  总集合中的下标
    sample_loader = wrap_data(data, targets, batch_size, root, dataset)

    return sample_loader, omega,samples_omega_index


def read_file(i, path):
    return i, Image.open(path).convert('RGB')


def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            self.onehot_targets = self.targets

            # 图片预加载，这样能像cifar10那样存在内存里，提高训练速度
            if self.dataset == 'cifar-10':
                #正常读取
                self.img = [0]*len(self.data)
                for i in range(len(self.data)):
                    self.img[i] = Image.fromarray(self.data[i])

            elif self.dataset == 'imagenet-100':
                #多线程+线程池读取
                self.img = [0]*len(self.data)
                with ThreadPoolExecutor(20) as executor:  # 创建 ThreadPoolExecutor，设置20个线程
                    future_list = [executor.submit(read_file, i,path) for i,path in enumerate(self.data)]  # 提交任务

                for future in as_completed(future_list):
                    i,result = future.result()  # 获取任务结果，为什么要i？，因为多线程是无序的，用i来标识图片的位置
                    self.img[i]=result
            else:
                raise ValueError('Invalid dataset name!')


        def __getitem__(self, index):
            img = self.transform(self.img[index])
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.targets)

    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    return dataloader


if __name__ == '__main__':
    #get_omegas(59000,10,7,0.5)
    query_dataloader, seen_mark_dataloader, seen_unmark_dataloader, unseen_mark_dataloader, unseen_unmark_dataloader, retrieval_dataloader = \
        load_data("cifar-10", "../datasets/cifar-10/", 7, 0.5, 8, 4)

    for img,target in query_dataloader:
        pass