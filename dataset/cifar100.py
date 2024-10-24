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
#from scipy import sparse
from helper.util import is_sorted

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
    "tin": 200,
}

DATASET_SAMPLES = {
    "cifar10": 50000,
    "cifar100": 50000,
    "tin": 100000,
}


# TIN: TinyImageNet
class TinInstance(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform): #, subset=200):
        self.transform = transform
        data_path = '%s/tinyImageNet200/%s/' % (data_folder, 'train')

        self.classes = sorted([basename(normpath(clss)) for clss in glob(data_path + '*/')])
        #total_cls = len(labels)
        #subset = np.clip(subset, a_min=0, a_max=total_cls)

        #self.img_list = [f for clss in labels[:subset] 
        self.img_list = [f for clss in self.classes
                         for f in glob(data_path + "/%s/images/*.JPEG" % (clss))]
        #self.img_list = [f for f in glob(data_path + "**/*.JPEG", recursive=True)]
        self.img_list.sort()

        self.num_classes = len(self.classes)
        self.name = 'tin'
        print("original ood dataset size: ", len(self.img_list))
        

    def set_index(self, indexes=None):
        if indexes is not None:
            self.img_list = np.array(self.img_list)[indexes]    #self.img_list[indexes]
        else:
            self.img_list = self.img_list
        print("cropped ood dataset size: ", len(self.img_list))

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
        if indexs is None:
            self.classes = np.unique(self.targets)
            self.num_classes = len(self.classes)
        else:
            self.data = self.data[indexs]
            real_targets = np.array(self.targets)[indexs]

            self.classes = np.unique(real_targets)
            self.num_classes = len(self.classes)

            assert is_sorted(real_targets) and is_sorted(self.classes), "Class idx need to be in sorted order"
            cls_pos = np.searchsorted(real_targets, self.classes)
            num_instance_cls = np.concatenate([cls_pos[1:]-cls_pos[:-1], len(real_targets)-cls_pos[-1:]])
            self.targets = np.arange(len(self.classes)).repeat(num_instance_cls)

            # NOTE: Assuming a chunk-based splitting mechanism 
            #instance_per_cls = len(real_targets)//self.num_classes
            #self.targets = np.arange(self.num_classes).repeat(instance_per_cls)

            assert np.all(real_targets == self.classes[self.targets]), "Incorrect mapping"


    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
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
        
    def __len__(self):
        return len(self.data)
    


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
        

def split_class_set(num_classes, split_size, rng=None, sorted=True):
    split_size = np.clip(split_size, a_min=0, a_max=num_classes)
    class_id = np.arange(num_classes)
    if rng is not None:
        rng.shuffle(class_id)
        
    split_a, split_b = class_id[:split_size], class_id[split_size:]
    if sorted:
        split_a.sort(); split_b.sort()
    return split_a, split_b


def get_class_idx(num_classes, num_full_class, split_seed):
    num_classes = np.clip(num_classes, a_min=0, a_max=num_full_class)
    assert num_classes > 0, "Number of class should be positive integer"

    class_idx = None      # Default no subset splitting; indicating full set
    if num_classes <  num_full_class:
        if split_seed is not None:
            split_rng = np.random.default_rng(seed=split_seed)
        
        class_idx, _ = split_class_set(num_full_class, num_classes, rng=split_rng)

    return class_idx



def x_u_split(base_dataset, 
              label_per_cls, instance_per_cls, include_labeled=True, 
              class_idx=None, include_unseen=False, 
              #lb_prop=1.0, #instance_prop=1.0, 
              min_size=0, rng=None):
    # Base dataset statistics
    N_SAMPLES = len(base_dataset)
    NUM_FULL_CLASS = base_dataset.num_classes 
    FULL_LB_PER_CLS = N_SAMPLES // NUM_FULL_CLASS
    
    #lb_prop = np.clip(lb_prop, 0, 1.0)
    #if instance_per_cls is not None:
    instance_per_cls = np.clip(instance_per_cls, min_size, FULL_LB_PER_CLS)
    label_per_cls = np.clip(label_per_cls, min_size, instance_per_cls)
    
    if class_idx is not None:
        assert np.all(class_idx < NUM_FULL_CLASS) and np.all(class_idx >= 0)
        assert len(np.unique(class_idx)) == len(class_idx), "Duplicate label involved"

    #label_per_class = int(np.ceil(lb_prop * full_label_per_class))
    #instance_per_class = int(np.ceil(instance_prop * full_label_per_class))
    #label_per_class = int(np.ceil(lb_prop * instance_per_class))
    labels = np.array(base_dataset.targets)

    
    labeled_idx = []
    unlabeled_idx = []
    for i in range(NUM_FULL_CLASS):
        idx = np.where(labels == i)[0]
        if rng is not None:     # Keep rng to ensure reproducing same split between teacher and student
            rng.shuffle(idx)    # Randomness of splitting

        idx = idx[:instance_per_cls]
        idx_u = idx.copy()
        idx_l = idx.copy() 
        if label_per_cls < instance_per_cls:
            idx_l = idx_l[:label_per_cls]
            idx_u = idx if include_labeled else idx[label_per_cls:]
        
        labeled_idx.append(idx_l)
        unlabeled_idx.append(idx_u)

    labeled_idx = np.stack(labeled_idx)
    if class_idx is not None:
        labeled_idx = labeled_idx[class_idx]
    labeled_idx = labeled_idx.flatten()

    #if not (lb_prop < 1.0 or include_labeled) \
    if not (label_per_cls < instance_per_cls or include_labeled) \
        and (class_idx is None or not include_unseen):   
        # Unlabeled set does not exist at all
        return labeled_idx, None
    
    unlabeled_idx = np.stack(unlabeled_idx)
    if not (class_idx is None or include_unseen):
        unlabeled_idx = unlabeled_idx[class_idx]
    unlabeled_idx = unlabeled_idx.flatten() 

    num_cls = NUM_FULL_CLASS if class_idx is None else len(class_idx)
    assert len(labeled_idx) == label_per_cls * num_cls \
        and (len(labeled_idx) + len(unlabeled_idx) == instance_per_cls * num_cls or include_labeled) \
        and (len(unlabeled_idx) == instance_per_cls * num_cls or not include_labeled)

    assert (len(labeled_idx) + len(unlabeled_idx) == N_SAMPLES or not include_unseen) 

    # Keep a minimum size by repeating the existing data
    min_size = np.clip(min_size, a_min=0, a_max=N_SAMPLES) 
    if len(labeled_idx) < min_size:
        labeled_idx = np.concatenate([labeled_idx for _ in range(int(np.ceil(min_size / len(labeled_idx))))])
    if len(unlabeled_idx) < min_size:
        unlabeled_idx = np.concatenate([unlabeled_idx for _ in range(int(np.ceil(min_size / len(unlabeled_idx))))])
    #np.random.shuffle(labeled_idx)
    print("Num seen class: ", num_cls)
    print("Num labeled: ", len(labeled_idx))
    print("Num unlabeled: ", 0 if unlabeled_idx is None else len(unlabeled_idx))

    return labeled_idx, unlabeled_idx



def split_ood_set(base_ood_set, num_subset_cls=200, samples_per_cls=None, #instance_prop=1.0, 
                  instance_split_seed=None, class_split_seed=None):

    num_full_cls = DATASET_CLASS[base_ood_set.name]
    num_subset_cls = np.clip(num_subset_cls, 0, num_full_cls)
    assert num_subset_cls > 0
    cls_idx = get_class_idx(num_subset_cls, num_full_cls, class_split_seed)

    # NOTE: Assuming balanced class samples
    FULL_SAMPLES_PER_CLS = len(base_ood_set) // base_ood_set.num_classes
    if samples_per_cls is None:
        samples_per_cls = FULL_SAMPLES_PER_CLS
    samples_per_cls = np.clip(samples_per_cls, 0, FULL_SAMPLES_PER_CLS)

    if instance_split_seed is not None:
        rng = np.random.default_rng(instance_split_seed)

    all_sample_idx = []
    for cls_id in range(base_ood_set.num_classes):  # TODO: Consider store & get samples for each cls from datafolder
        sample_idx = np.arange(FULL_SAMPLES_PER_CLS) + cls_id * samples_per_cls
        #begin_idx, end_idx = sample_idx_per_cls[cls_id-1], sample_idx_per_cls[cls_id]
        #sample_idx = np.arange(begin_idx, end_idx)
        if instance_split_seed is not None:
            rng.shuffle(sample_idx)

        sample_idx = sample_idx[:samples_per_cls]
        all_sample_idx.append(sample_idx)

    all_sample_idx = np.stack(all_sample_idx)
    if cls_idx is not None:
        all_sample_idx = all_sample_idx[cls_idx]
    return all_sample_idx.flatten()


def get_cifar100_test(batch_size=64, num_workers=4, 
                      num_classes=DATASET_CLASS['cifar100'],
                      num_samples=DATASET_SAMPLES['cifar100'], 
                      split_seed=None):
    data_folder = get_data_folder()
    
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    full_test_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    
    if num_classes < DATASET_CLASS['cifar100']:
        class_idx = get_class_idx(num_classes, num_full_class=DATASET_CLASS['cifar100'], split_seed=split_seed)
        samples_per_cls = num_samples //num_classes

        test_idx, _ = x_u_split(full_test_set, 
                                instance_per_cls=samples_per_cls, label_per_cls=samples_per_cls,
                                class_idx=class_idx, include_unseen=False)
    else:
        test_idx = None
    
    test_set = CIFAR100Instance(root=data_folder + '/cifar/',
                                download=True,
                                train=False,
                                transform=test_transform, 
                                indexs=test_idx)
    
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
        
    return test_loader  
    


def get_cifar100_dataloaders(batch_size=128, num_workers=8, 
                             num_id_class=DATASET_CLASS['cifar100'], 
                             ood='tin', num_ood_class=DATASET_CLASS['tin'], 
                             num_samples=DATASET_SAMPLES['cifar100'], 
                             num_labels=DATASET_SAMPLES['cifar100'],
                             #lb_prop=1.0, 
                             label_per_cls=None, 
                             include_labeled=False, 
                             split_seed=None, class_split_seed=None):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])


    base_dataset = CIFAR100Instance(root=data_folder + '/cifar/',
                                     download=True,
                                     train=True,
                                     transform=train_transform)

    NUM_FULL_CLASS, NUM_FULL_OOD_CLS = DATASET_CLASS['cifar100'], DATASET_CLASS[ood]
    num_id_class = np.clip(num_id_class, 1, NUM_FULL_CLASS)
    num_ood_class = np.clip(num_ood_class, 0, NUM_FULL_OOD_CLS)
    
    num_samples = np.clip(0, num_samples, min(DATASET_SAMPLES['cifar100'], DATASET_SAMPLES[ood]))
    num_labels = np.clip(0, num_labels, num_samples)

    # Fully supervied training
    #if not lb_prop < 1.0 and not include_labeled \
    if not (num_id_class < NUM_FULL_CLASS or num_samples < DATASET_SAMPLES['cifar100'] 
        or num_labels < num_samples or include_labeled \
        or num_ood_class > 0):
        train_loader = DataLoader(base_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)
        return train_loader, None

    # Split dataset given requested num_id_class
    class_idx = get_class_idx(num_id_class, num_full_class=NUM_FULL_CLASS, split_seed=class_split_seed) \
            if num_id_class < NUM_FULL_CLASS else None

    samples_per_cls = num_samples //(num_id_class + num_ood_class)
    label_per_cls = num_labels //(num_id_class)


    lb_idx, ulb_idx = x_u_split(base_dataset, 
                                instance_per_cls=samples_per_cls, label_per_cls=label_per_cls,
                                #instance_prop=instance_prop, lb_prop=lb_prop, 
                                include_labeled=include_labeled, 
                                class_idx=class_idx, include_unseen=False, 
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
        return train_loader, utrain_loader
    
    
    if ood == 'tin':
        utrain_transform = transforms.Compose([
            transforms.Resize(32),
            # transforms.RandomCrop(32, padding=4), # TODO
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        ood_set = TinInstance(
            data_folder, transform=utrain_transform)    #, subset=num_ood_class)
        # Requested dataset splitting
        sample_idx = split_ood_set(ood_set, num_subset_cls=num_ood_class, samples_per_cls=samples_per_cls, #instance_prop=instance_prop,
                                    instance_split_seed=split_seed, class_split_seed=class_split_seed)
        ood_set.set_index(sample_idx)

    elif ood == 'places':
        utrain_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])
        ood_set = Places365Instance(
            data_folder + '/places365', transform=utrain_transform)
    else:
        assert False, "OOD dataset Not Implemented"


    if ulb_train_set is None:
        #mu = len(ood_set)//len(lb_train_set)
        utrain_loader = DataLoader(ood_set, batch_size=batch_size, #* mu,
                            shuffle=True, num_workers=num_workers, drop_last=True) 
        print("unlabeled datasize: ", len(ood_set))
    else:
        unlabeled_set = MixedDataset(ulb_train_set, ood_set)
        #mu = len(unlabeled_set)//len(lb_train_set)
        utrain_loader = DataLoader(unlabeled_set, batch_size=batch_size, #* mu,
                                shuffle=True, num_workers=num_workers, drop_last=True)
        print("unlabeled datasize: ", len(unlabeled_set))
    
    return train_loader, utrain_loader


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