import torch
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_cifar10_dataloader(root, batch_size, num_workers,omegas):
    """
    Load cifar10 dataset.
    Args
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.
    Returns
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    CIFAR10.init(root,omegas)
    query_dataset = CIFAR10('query', transform=query_transform())
    seen_mark_dataset = CIFAR10('seen_mark', transform=train_transform())
    seen_unmark_dataset = CIFAR10('seen_unmark', transform=train_transform())
    unseen_mark_dataset = CIFAR10('unseen_mark', transform=train_transform())
    unseen_unmark_dataset = CIFAR10('unseen_unmark', transform=train_transform())
    retrieval_dataset = CIFAR10('retrieval', transform=train_transform())

    query_dataloader = DataLoader(query_dataset,batch_size=batch_size,pin_memory=True,num_workers=num_workers)

    seen_mark_dataloader = DataLoader(seen_mark_dataset,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=num_workers)
    seen_unmark_dataloader = DataLoader(seen_unmark_dataset,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=num_workers)

    unseen_mark_dataloader = DataLoader(unseen_mark_dataset,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=num_workers)
    unseen_unmark_dataloader = DataLoader(unseen_unmark_dataset,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=num_workers)

    retrieval_dataloader = DataLoader(retrieval_dataset,shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=num_workers)

    return query_dataloader, seen_mark_dataloader, seen_unmark_dataloader,unseen_mark_dataloader,unseen_unmark_dataloader, retrieval_dataloader

class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root,omegas):
        # Load data
        # 注意，这个数据集是有序的,所以下面生成的omega是根据有序数组生成的随便下标
        CIFAR10.QUERY_DATA = np.load(os.path.join(root, 'cifar10_1000_query_data.npy'))
        CIFAR10.QUERY_TARGETS = np.load(os.path.join(root, 'cifar10_1000_query_onehot_targets.npy'))
        CIFAR10.RETRIEVAL_DATA = np.load(os.path.join(root, 'cifar10_59000_retrieval_data.npy'))
        CIFAR10.RETRIEVAL_TARGETS = np.load(os.path.join(root, 'cifar10_59000_retrieval_onehot_targets.npy'))

        # train_data_len = len(CIFAR10.RETRIEVAL_DATA)
        # class_num = 10

        train_seen_mark_omega, train_seen_unmark_omega, train_unseen_mark_omega, train_unseen_unmark_omega=omegas

        CIFAR10.SEEN_MARK_DATA = CIFAR10.RETRIEVAL_DATA[train_seen_mark_omega]
        CIFAR10.SEEN_MARK_TARGETS = CIFAR10.RETRIEVAL_TARGETS[train_seen_mark_omega]
        CIFAR10.SEEN_MARK_OMEGA = train_seen_mark_omega

        CIFAR10.SEEN_UNMARK_DATA = CIFAR10.RETRIEVAL_DATA[train_seen_unmark_omega]
        CIFAR10.SEEN_UNMARK_TARGETS = CIFAR10.RETRIEVAL_TARGETS[train_seen_unmark_omega]
        CIFAR10.SEEN_UNMARK_OMEGA = train_seen_unmark_omega

        CIFAR10.UNSEEN_MARK_DATA = CIFAR10.RETRIEVAL_DATA[train_unseen_mark_omega]
        CIFAR10.UNSEEN_MARK_TARGETS = CIFAR10.RETRIEVAL_TARGETS[train_unseen_mark_omega]
        CIFAR10.UNSEEN_MARK_OMEGA = train_unseen_mark_omega

        CIFAR10.UNSEEN_UNMARK_DATA = CIFAR10.RETRIEVAL_DATA[train_unseen_unmark_omega]
        CIFAR10.UNSEEN_UNMARK_TARGETS = CIFAR10.RETRIEVAL_TARGETS[train_unseen_unmark_omega]
        CIFAR10.UNSEEN_UNMARK_OMEGA = train_unseen_unmark_omega

    def __init__(self, mode,
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'seen_mark':
            self.data = CIFAR10.SEEN_MARK_DATA
            self.targets = CIFAR10.SEEN_MARK_TARGETS
            self.omega = CIFAR10.SEEN_MARK_OMEGA
        elif mode == 'seen_unmark':
            self.data = CIFAR10.SEEN_UNMARK_DATA
            self.targets = CIFAR10.SEEN_UNMARK_TARGETS
            self.omega = CIFAR10.SEEN_UNMARK_OMEGA
        elif mode == 'unseen_mark':
            self.data = CIFAR10.UNSEEN_MARK_DATA
            self.targets = CIFAR10.UNSEEN_MARK_TARGETS
            self.omega = CIFAR10.UNSEEN_MARK_OMEGA
        elif mode == 'unseen_unmark':
            self.data = CIFAR10.UNSEEN_UNMARK_DATA
            self.targets = CIFAR10.UNSEEN_UNMARK_TARGETS
            self.omega = CIFAR10.UNSEEN_UNMARK_OMEGA

        elif mode == 'query':
            self.data = CIFAR10.QUERY_DATA
            self.targets = CIFAR10.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = CIFAR10.RETRIEVAL_DATA
            self.targets = CIFAR10.RETRIEVAL_TARGETS
        else:
            raise ValueError('Mode error!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.targets)

    def get_omega(self):
        return torch.from_numpy(self.omega)