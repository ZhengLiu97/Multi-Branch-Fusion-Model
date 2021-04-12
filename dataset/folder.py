from torch.utils import data
import sys
import os


class FolderDataset(data.Dataset):
    """
    工具类: 用于总目录中，每个目录对应一种类别的数据集
    Args:
        root: 总目录的根目录
        loader: 一个函数，定义如何加载数据集中的文件,输入一个文件路径,返回一个向量
        in_ram: 是否把所有数据均加载到内存中
    """
    def __init__(self, root, loader, in_ram=False, target_transform=None):
        self.loader = loader
        self.root = root
        self.in_ram = in_ram
        self.target_transform = target_transform
        classes, class_to_idx = FolderDataset._find_classes(self.root)
        self.path_with_idx = FolderDataset._get_path_with_idx(self.root, class_to_idx)
        if self.in_ram:
            self.data = FolderDataset._load_dataset(self.path_with_idx, self.loader)

    @staticmethod
    def _find_classes(tdir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(tdir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def _get_path_with_idx(tdir, class_to_idx):
        ans = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(tdir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    ans.append((path, class_to_idx[target]))
        return ans

    @staticmethod
    def _load_dataset(path_with_idx, loader):
        ans = []
        for path, _ in path_with_idx:
            ans.append(loader(path))
        return ans

    def __getitem__(self, index):
        target = self.target_transform(self.path_with_idx[index][1])
        if self.in_ram:
            return self.data[index], target
        else:
            return self.loader(self.path_with_idx[index][0]), target

    def __len__(self):
        return len(self.path_with_idx)
