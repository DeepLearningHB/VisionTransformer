import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import tarfile
from dataset.pair_loader import *

class CUB2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.classes = 200
        self.loader = loader
        if download:
            self.download()
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.targets = [self.data.iloc[i].target - 1 for i in range(len(self.data))]

    def load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def check_integrity(self):
        try:
            self.load_metadata()
        except Exception:
            return False
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def download(self):
        if self.check_integrity():
            print('Files already downloaded and verified')
            return
        download_file_from_google_drive(self.url, self.root, self.filename, self.tgz_md5)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, idx

std_value = 1.0 / 255.0
normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])



def get_cub_dataloader(batch_size=128, num_workers=8, is_instance=False):
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
    cub_train = CUB2011('/home/leehanbeen/cub_200', True, train_transform, download=False)
    train_loader = DataLoader(cub_train, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=True)
    cub_test = CUB2011('/home/leehanbeen/cub_200', False, test_transform, download=False)
    test_loader = DataLoader(cub_test, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(cub_train)
    return train_loader, test_loader

def get_cub_cskd_loader(batch_size=128, num_workers=8, is_instance=False):
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

    cub_train = CUB2011('/home/leehanbeen/cub_200', True, train_transform, download=False)
    cub_test = CUB2011('/home/leehanbeen/cub_200', False, test_transform, download=False)
    cub_train = DatasetWrapper(cub_train)
    cub_test = DatasetWrapper(cub_test)
    from torch.utils.data import DataLoader
    get_train_sampler = lambda d: PairBatchSampler(d, batch_size)
    get_test_sampler = lambda d: BatchSampler(SequentialSampler(d), batch_size, False)

    train_loader = DataLoader(cub_train, batch_sampler=get_train_sampler(cub_train), num_workers=num_workers)
    test_loader = DataLoader(cub_test, batch_sampler=get_test_sampler(cub_test), num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(cub_train)
    return train_loader, test_loader

if __name__ == "__main__":
    a, b = get_cub_cskd_loader(batch_size=32)

    for i in a:
        print(i[0].shape, i[1].shape)
        break
    for i in b:
        print(i[0].shape, i[1].shape)
        break


