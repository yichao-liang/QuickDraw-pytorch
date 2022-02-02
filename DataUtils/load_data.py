import os
import numpy as np
import torch
import torch.utils.data as data


def load_dataset(root, mtype):
    num_classes = 0
    with open("./data/QuickDraw_pytorch/DataUtils/class_names.txt", "r") as f:
        for line in f:
            num_classes = num_classes+1

    # load data from cache
    if os.path.exists(os.path.join(root, mtype+'.npz')):
        print("*"*50)
        print("Loading "+mtype+" dataset...")
        print("*"*50)
        print("Classes number of "+mtype+" dataset: "+str(num_classes))
        print("*"*50)
        data_cache = np.load(os.path.join(root, mtype+'.npz'))
        return data_cache["data"].astype('float32'), \
            data_cache["target"].astype('int64'), num_classes

    else:
        raise FileNotFoundError("%s doesn't exist!" %
                                os.path.join(root, mtype+'.npz'))


class QD_Dataset(data.Dataset):
    def __init__(self, mtype, transform, root='Dataset'):
        """
        args:
        - mytpe: str, specify the type of the dataset, i.e, 'train' or 'test'
        - root: str, specify the root of the dataset directory
        """

        self.data, self.target, self.num_classes = load_dataset(root, mtype)
        self.data = self.data 
        self.target = torch.from_numpy(self.target)
        self.transform = transform
        print("Dataset "+mtype+" loading done.")
        print("*"*50+"\n")

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape(28,28,1) / 255
        return self.transform(img).view(1, 50, 50), self.target[index]

    def __len__(self):
        return len(self.data)

    def get_number_classes(self):
        return self.num_classes