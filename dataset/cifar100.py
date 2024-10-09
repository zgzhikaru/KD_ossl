from __future__ import print_function

import cv2
import numpy as np
import os
import socket
import torch
from PIL import Image
from glob import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils import get_data_folder, TransformTwice
from os.path import basename, normpath

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

DATASET_CLASS = {
    "cifar10": 10,
    "cifar100": 100,
}

# TIN: TinyImageNet
class TinInstance(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform, subset=200):
        self.transform = transform
        data_path = '%s/tinyImageNet200/%s/' % (data_folder, 'train')

        labels = sorted([basename(normpath(clss)) for clss in glob(data_path + '*/')])
        total_cls = len(labels)

        subset = np.clip(subset, a_min=0, a_max=total_cls)
        if not subset:  # Requesting zero-size dataset indicating no loader needed
            self.img_list = []
            return

        self.img_list = [f for clss in labels[:subset] 
                         for f in glob(data_path + "/%s/images/*.JPEG" % (clss))]
        #self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        self.img_list.sort()
        print("ood dataset size: ", len(self.img_list))
        

    def set_index(self, indexes=None):
        if indexes is not None:
            self.img_list = self.img_list[indexes]
        else:
            self.img_list = self.img_list

    def init_index(self):
        self.img_list = self.img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = img.convert('RGB')
        return self.transform(img), index


class Places365Instance(datasets.Places365):
    """places365Instance Dataset.
    """

    def __getitem__(self, index):
        file, target = self.imgs[index]
        image = self.loader(file)
        return self.transform(image), index


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __init__(self, root, train, download, transform, target_transform=None, unlabeled=False, 
                    indexs=None):    #lb_prop=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        #assert lb_prop > 0 and lb_prop <= 1.0, "Invalid proportion"
        self.unlabeled = unlabeled
        self.num_classes = len(np.unique(self.targets))
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

            self.num_classes = len(np.unique(self.targets))


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.unlabeled:
            return img, index
        else:
            return img, target, index


class InstanceSample(torch.utils.data.Dataset):
    """
    Instance+Sample Dataset
    """

    def __init__(self, root, data, model, transform, target_transform=None, k=4096, mode='exact', is_sample=True,
                 percent=1.0):

        if data == 'tin':
            infos = open('%s/tinyImageNet200/words.txt' %
                         (root)).readlines()
        else:
            raise NotImplementedError

        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        for info in infos:
            img_name, img_label = info.strip('\n').split(';')
            # TODO: Make data_path a input configuration; Otherwise make it less path specific
            img = cv2.imread(
                '/data/Datasets/classification/' + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.data.append(img)
            self.targets.append(int(img_label)) # FIXME: TIN gt-label index collide with cifar labels?; TIN gt is assumed no access under SSL setting

        cifar100_data = datasets.CIFAR100(
            root+'/cifar/', train=True, download=True)

        self.data.extend(list(cifar100_data.data))      # Merging two datasets
        self.targets.extend(list(cifar100_data.targets))

        print(len(self.data), len(self.targets))

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100   # FIXME: num_classes = len(cifar100.classes) + len(tin.classes) = 300
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)   # WARNING: Indexing out-of-bound error if cls_pos includes only first 100 classes

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i])
                             for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i])
                             for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(
                self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i])
                             for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i])
                             for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(
                self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def x_u_split(base_dataset, lb_prop=1.0, include_labeled=True, 
              num_unseen_class=0, include_unseen=False, 
              min_size=0, rng=None):
    
    n_data = len(base_dataset)
    num_classes = base_dataset.num_classes  #len(np.unique(labels))
    full_label_per_class = n_data // num_classes
    
    lb_prop = np.clip(lb_prop, 0, 1.0)
    num_unseen_class = np.clip(num_unseen_class, 0, num_classes).astype(int)

    unlabeled_idx = np.arange(n_data) 
    if not num_unseen_class > 0 and not lb_prop < 1.0 and not num_unseen_class > 0:  # No split requested
        labeled_idx = np.arange(n_data)
        #if not include_labeled:
        return labeled_idx, unlabeled_idx if include_labeled else None
    
    
    if num_unseen_class == num_classes: # Fully remove all classes
        return None, None
    elif num_unseen_class > 0:    # Remove unseen classes
        num_classes -= num_unseen_class

    #num_labeled = np.ceil(n_data * lb_prop)     # Compute num of labeled data
    #label_per_class = num_labeled // num_classes
    label_per_class = int(np.ceil(lb_prop * full_label_per_class))
    labels = np.array(base_dataset.targets)

    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx_l = idx.copy() 
        if lb_prop < 1.0:
            if rng is None:
                print("Deprecated: RNG is not set; Unsaved split cannot be retrieved for downstream-tasks")
                idx_l = np.random.choice(idx_l, label_per_class, False)
            else:
                idx_l = rng.choice(idx_l, label_per_class, False)   
        # Keep rng to ensure reproducing same split between teacher and student
        labeled_idx.extend(idx_l)
        unlabeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx) if not include_unseen else np.arange(n_data)

    assert len(labeled_idx) == label_per_class * num_classes \
        and (len(unlabeled_idx) == full_label_per_class * num_classes and not include_unseen)

    if not include_labeled:
        if lb_prop < 1.0:
            unlabeled_idx = np.setdiff1d(unlabeled_idx, labeled_idx)
            assert (len(labeled_idx) + len(unlabeled_idx) == n_data and include_unseen) \
                or (len(labeled_idx) + len(unlabeled_idx) == full_label_per_class * num_classes and not include_unseen)
        else:   # 
            return labeled_idx, None

    min_size = np.clip(min_size, a_min=0, a_max=n_data) # TODO: Move the min_size check before each return
    if len(labeled_idx) < min_size:
        labeled_idx = np.concatenate([labeled_idx for _ in range(int(np.ceil(min_size / len(labeled_idx))))])
    if len(unlabeled_idx) < min_size:
        unlabeled_idx = np.concatenate([unlabeled_idx for _ in range(int(np.ceil(min_size / len(unlabeled_idx))))])
    #np.random.shuffle(labeled_idx)
    print("Num seen class: ", num_classes)
    print("Num labeled: ", len(labeled_idx))
    print("Num unlabeled: ", 0 if unlabeled_idx is None else len(unlabeled_idx))
    return labeled_idx, unlabeled_idx


def get_unseen_class(num_id_class, num_ood_class, num_total_class=None):
    #num_id_class = base_dataset.num_classes
    if num_total_class is None:
        return 0
    #num_unseen_class = 0
    #if num_total_class is not None:     # Constraint on total number of classes
    #num_total_class = num_id_class  # Then: num_ood > 0 & < num_id
    
    assert  num_total_class > num_ood_class \
        and num_total_class <=  num_ood_class + num_id_class, \
        "Range Exceeded: Number of specified total class "
    
    num_unseen_class = num_id_class - num_total_class + num_ood_class
    #num_id_class - (num_total_class - num_ood_class)
    # NOTE: Can be extended to request extra num_unseen_class by adding another argument

    return num_unseen_class


def get_cifar100_dataloaders(batch_size=128, num_workers=8,
                             is_instance=False,
                             is_sample=True,
                             k=4096, mode='exact', percent=1.0, ood='tin', model=None, 
                             lb_prop=1.0, include_labeled=False, 
                             num_ood_class=200, num_total_class=None, 
                             split_seed=None):
    """
    cifar 100
    """
    if is_instance:
        assert not is_sample
    if is_sample:
        assert not is_instance

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    tiny_transform = transforms.Compose([
        transforms.Resize(32),
        # transforms.RandomCrop(32, padding=4), # TODO
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    if is_instance and not is_sample:
        base_dataset = CIFAR100Instance(root=data_folder + '/cifar/',
                                     download=True,
                                     train=True,
                                     transform=train_transform)

    elif is_sample and not is_instance:
        base_dataset = InstanceSample(root=data_folder, data=ood, model=model, transform=tiny_transform,
                                   target_transform=None, k=k, mode=mode, is_sample=True, percent=percent)
    else:
        base_dataset = datasets.CIFAR100(root=data_folder + '/cifar/',
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    n_data = len(base_dataset)

    train_loader = DataLoader(base_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    # Split test set given num_unseen_class
    num_unseen_class = get_unseen_class(base_dataset.num_classes, num_ood_class, num_total_class=num_total_class)
    """
    test_set = datasets.CIFAR100(root=data_folder + '/cifar/',
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    """
    test_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    if num_unseen_class > 0:
        test_idx, _ = x_u_split(test_set, 
                                num_unseen_class=num_unseen_class, include_unseen=False, 
                                rng=np.random.default_rng(split_seed))
        test_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                    download=True,
                                    train=False,
                                    transform=test_transform, 
                                    indexs=test_idx)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))
    

    # Fully supervied training
    if not lb_prop < 1.0 and not num_ood_class > 0 and not include_labeled:
            train_loader = DataLoader(base_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=True)
            return train_loader, None, test_loader, n_data
        
    
 
    lb_idx, ulb_idx = x_u_split(base_dataset, 
                                lb_prop=lb_prop, include_labeled=include_labeled, 
                                num_unseen_class=num_unseen_class, include_unseen=False, 
                                min_size=batch_size, 
                                rng=np.random.default_rng(split_seed))
    
    lb_train_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                    download=True,
                                    train=True,
                                    transform=train_transform, 
                                    indexs=lb_idx)

    ulb_train_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                    download=True,
                                    train=True,
                                    transform=train_transform, 
                                    unlabeled=True, 
                                    indexs=ulb_idx) if ulb_idx is not None else None

    train_loader = DataLoader(lb_train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)
    print("num seen class: ", lb_train_set.num_classes)
    print("labeled datasize: ", len(lb_train_set))

    if not num_ood_class > 0:   # SSL with only ID data
        utrain_loader = DataLoader(ulb_train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True) if ulb_train_set is not None else None
        return train_loader, utrain_loader, test_loader, n_data
    
    
    if ood == 'tin':
        ood_set = TinInstance(
            data_folder, transform=tiny_transform, subset=num_ood_class)
    elif ood == 'places':
        ood_set = Places365Instance(
            data_folder + '/places365', transform=tiny_transform)

    if ulb_train_set is None:
        utrain_loader = DataLoader(ood_set, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, drop_last=True) 
        print("unlabeled datasize: ", len(ood_set))
    else:
        unlabeled_set = MixedDataset(ulb_train_set, ood_set)
        utrain_loader = DataLoader(unlabeled_set, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, drop_last=True)
        print("unlabeled datasize: ", len(unlabeled_set))
    
    if is_instance or is_sample:
        return train_loader, utrain_loader, test_loader, n_data
    else:
        return train_loader, utrain_loader, test_loader


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, id_set, ood_set):
        self.id_set = id_set
        self.ood_set = ood_set

        self.id_size = len(id_set)
        self.ood_size = len(ood_set)
        

    def __len__(self):
        return self.id_size + self.ood_size

    def __getitem__(self, index):
        if index < self.id_size:
            return self.id_set[index]
        elif index < self.id_size + self.ood_size:
            return self.ood_set[index - self.id_size]
        else:
            assert False, "Wrong indexing or sizing"