import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


class VoronoiRegionDataset(Dataset):

    def __init__(self, root,limit=None):
        assert os.path.exists(root)
        LABELS_PATH = os.path.join(root, "labels.csv")
        self.labels = pd.read_csv(LABELS_PATH)
        if limit is not None:
            self.labels=self.labels.iloc[:limit]
        self.root = root
        self.toTensor = transforms.ToTensor()
        self.images_df = None
       # self.load_images()


    def __len__(self):
        return len(self.labels)

    def preprocess(self, img):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(img)

    def __get_item_path__(self, index):
        return os.path.join(self.root, self.labels.iloc[index, 1])

    def __get_num_classes__(self):
        classes = self.labels.label.unique()
        return len(classes)

    def load_images(self):
        if self.images_df is None:
            self.images_df = pd.DataFrame()
            for image_index in tqdm(range(self.__len__()),'loading images'):
                image, label = self.load_image(image_index,True)
                self.images_df=self.images_df.append({'image_arr':image,'label':label},ignore_index=True)
        


    def load_image(self,index, transform=True):
        path = self.__get_item_path__(index)
        label = torch.tensor(int(self.labels.iloc[index, 2])) - 1
        with Image.open(path) as img:
            if transform:
                img = self.preprocess(img)
                # img = img.unsqueeze(0)
                return img, label
            else:
                return img, label

    def __getitem__(self, index):

        #label = self.images_df.iloc[index]['label']
        #img = self.images_df.iloc[index]['image_arr']
        img, label = self.load_image(index,True)
        return img, label

        #return img, label

    def plot(self, index):
        path = self.__get_item_path__(index)
        with Image.open(path) as img:
            img = self.preprocess(img)
            img = img.numpy()
            img = np.moveaxis(img, 0, -1)
            filename = self.labels.iloc[index, 1]
            fig = plt.figure()
            fig.suptitle(f'{filename}', fontsize=20)
            plt.imshow(img)
            plt.show()

    def __getlabels__(self):
        return self.labels
