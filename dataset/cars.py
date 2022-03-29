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

class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    file_list = {
        'imgs': ('http://ai.stanford.edu/~jkrause/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://ai.stanford.edu/~jkrause/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        self.classes = 196
        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        self.targets = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))
                self.targets.append(label)

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)

std_value = 1.0 / 255.0
normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])


def get_car_dataloader(batch_size=128, num_workers=8, is_instance=False):
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
    car_train = Cars('/home/leehanbeen/stanford_cars', train=True, transform=train_transform, download=False)
    train_loader = DataLoader(car_train, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=True)
    car_test = Cars('/home/leehanbeen/stanford_cars', train=False, transform=test_transform, download=False)

    test_loader = DataLoader(car_test, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(car_train)
    return train_loader, test_loader


def get_cars_cskd_loader(batch_size=128, num_workers=8, is_instance=False):
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

    car_train = Cars('/home/leehanbeen/stanford_cars', train=True, transform=train_transform, download=False)
    car_test =Cars('/home/leehanbeen/stanford_cars', train=False, transform=test_transform, download=False)
    car_train = DatasetWrapper(car_train)
    car_test = DatasetWrapper(car_test)
    from torch.utils.data import DataLoader
    get_train_sampler = lambda d: PairBatchSampler(d, batch_size)
    get_test_sampler = lambda d: BatchSampler(SequentialSampler(d), batch_size, False)

    train_loader = DataLoader(car_train, batch_sampler=get_train_sampler(car_train), num_workers=num_workers)
    test_loader = DataLoader(car_test, batch_sampler=get_test_sampler(car_test), num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, len(car_train)
    return train_loader, test_loader

if __name__ == "__main__":
    a, b = get_cars_cskd_loader(batch_size=32)

    for i in a:
        print(i[0].shape, i[1].shape)
        break
    for i in b:
        print(i[0].shape, i[1].shape)
        break
