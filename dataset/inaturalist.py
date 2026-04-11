import os
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

class INaturalist(Dataset):

    def __init__(self, root, txt, transform=None, train=True, class_balance=False, ordered_data=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.train = train
        self.class_balance = class_balance
        self.ordered_data = ordered_data


        with open(txt, 'r') as f:
            for line in f:

                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                path, label = parts[0], parts[1]

                self.img_path.append(os.path.join(root, path))
                self.labels.append(int(label))

        if len(self.labels) > 0:
            self.num_classes = len(np.unique(self.labels))
        else:
            self.num_classes = 0


        self.class_data = [[] for _ in range(self.num_classes)]
        for idx, lbl in enumerate(self.labels):
            self.class_data[lbl].append(idx)

        self.cls_num_list = [len(self.class_data[c]) for c in range(self.num_classes)]


        self.ordered_labels = []
        self.ordered_img_path = []
        self.classes_samples_pointers = {c: [] for c in range(self.num_classes)}
        start_idx = 0
        for c in range(self.num_classes):

            self.ordered_labels.extend([c] * len(self.class_data[c]))
            self.ordered_img_path.extend([self.img_path[i] for i in self.class_data[c]])
            end_idx = len(self.ordered_labels) - 1
            self.classes_samples_pointers[c] = [start_idx, end_idx]
            start_idx = end_idx + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
            path = self.img_path[index]

        elif self.ordered_data:
            path = self.ordered_img_path[index]
            label = self.ordered_labels[index]
        else:
            path = self.img_path[index]
            label = self.labels[index]


        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')


        if self.transform is not None:
            if self.train:

                if isinstance(self.transform, list) or isinstance(self.transform, tuple):

                    views = [trans(image) for trans in self.transform]
                    return views, label
                else:

                    return self.transform(image), label
            else:

                if isinstance(self.transform, list) or isinstance(self.transform, tuple):

                    return self.transform[0](image), label
                else:
                    return self.transform(image), label
        else:

            return image, label

    def get_cls_num_list(self):

        return self.cls_num_list
