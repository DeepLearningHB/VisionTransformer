import os
import scipy.io as sio
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import torchvision.transforms as transforms

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from dataset.pair_loader import *

from PIL import Image


class FGVC(Dataset):
    def __init__(self, label_file=None, transform=None):
        self.label_file = label_file
        file = open(self.label_file, 'r')
        self.readlines = file.readlines()
        self.data_list = []
        self.transform = transform
        self.classes = 100
        self.image_basepath = '/'.join(self.label_file.split("/")[:-1]) + "/images"

        for line in self.readlines:
            line = line.replace("\n", '')
            filename = line.split(' ')[0]+".jpg"
            label = ''.join(line.split(" ")[1:])
            self.data_list.append((filename, label))

        label_list = sorted(list(set([data[1] for data in self.data_list])))
        print(len(label_list))
        self.label_dict = {}
        for idx, label in enumerate(label_list):
            self.label_dict[label] = idx

        self.targets = [self.label_dict[data[1]] for data in self.data_list]
        print(self.targets)



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filename, label = self.data_list[idx]

        file_path = os.path.join(self.image_basepath, filename)
        img = Image.open(file_path)
        img = img.convert('RGB')
        ret_img = self.transform(img)
        ret_label = self.label_dict[label]
        return ret_img, ret_label, idx


std_value = 1.0 / 255.0
normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])

def get_fgvc_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    train_label_file = '/home/leehanbeen/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_trainval.txt'
    test_label_file = '/home/leehanbeen/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt'

    train_transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.RandomResizedCrop(scale=(0.16, 1), size=227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize,
    ])

    print(train_transform)
    from torch.utils.data import DataLoader
    fgvc_train = FGVC(train_label_file, train_transform)
    train_loader = DataLoader(fgvc_train, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=True)
    fgvc_test = FGVC(test_label_file, test_transform)
    test_loader = DataLoader(fgvc_test, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(fgvc_train)
    return train_loader, test_loader

def get_fgvc_cskd_loader(batch_size=128, num_workers=8, is_instance=False):
    train_label_file = '/home/leehanbeen/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_trainval.txt'
    test_label_file = '/home/leehanbeen/fgvc-aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt'

    train_transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.RandomResizedCrop(scale=(0.16, 1), size=227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize,
    ])

    fgvc_train =  FGVC(train_label_file, train_transform)
    fgvc_test =  FGVC(test_label_file, test_transform)
    fgvc_train = DatasetWrapper(fgvc_train)
    fgvc_test = DatasetWrapper(fgvc_test)
    from torch.utils.data import DataLoader
    get_train_sampler = lambda d: PairBatchSampler(d, batch_size)
    get_test_sampler = lambda d: BatchSampler(SequentialSampler(d), batch_size, False)

    train_loader = DataLoader(fgvc_train, batch_sampler=get_train_sampler(fgvc_train), num_workers=num_workers)
    test_loader = DataLoader(fgvc_test, batch_sampler=get_test_sampler(fgvc_test), num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(fgvc_train)
    return train_loader, test_loader

if __name__ == "__main__":
    a, b = get_fgvc_cskd_loader(batch_size=32)

    for i in a:
        print(i[0].shape, i[1].shape)
        break
    for i in b:
        print(i[0].shape, i[1].shape)
        break