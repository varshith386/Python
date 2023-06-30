import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "./data"
    testing_dir = "./data"
    train_batch_size = 64
    train_number_epochs = 100



class Data(Dataset):

    def __init__(self, face_dir, transform=None):
        self.face_dir = face_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_dir.imgs)

    def __getitem__(self, index):
        img_packed = self.face_dir.imgs[index]
        img = Image.open(img_packed[0])
        img = self.transform(img)
        label = img_packed[1]
        return img, label
    



if __name__ == "__main__":
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    TRANSFORM = transforms.Compose([transforms.Resize((300,300)),transforms.ToTensor()])
    
    siamese_dataset = Data(face_dir=folder_dataset,transform=TRANSFORM)

    print(f"There are {len(siamese_dataset)} samples in the dataset.")
    img,label = siamese_dataset[0]
    print(f"Shape of image: {img.shape}")
    print(label)