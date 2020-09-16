import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from crop import *
import cv2


def preprocess_crop(root='data/', img_size=(224,224)):
    no_image_files = os.listdir(os.path.join(root, "no"))
    yes_image_files = os.listdir(os.path.join(root, "yes"))
    no_image_files = [root + "no/" + s for s in no_image_files]
    yes_image_files = [root + "yes/" + s for s in yes_image_files]

    no_images = list(map(cv2.imread, no_image_files))
    no_images = [cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC) for img in no_images]

    yes_images = list(map(cv2.imread, yes_image_files))
    yes_images = [cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC) for img in yes_images]

    no_images = [cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC) for img in
                      crop_imgs(no_images)]
    yes_images = [cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC) for img in
                       crop_imgs(yes_images)]


    #whole = np.array(no_images + yes_images)
    #print(np.mean(whole, axis=(0,1,2)), np.std(whole, axis=(0,1,2)))
    for i, img in enumerate(no_images):
        cv2.imwrite(root + 'crop/no/'+str(i+1)+'.jpg', img)

    for i, img in enumerate(yes_images):
        cv2.imwrite(root + 'crop/yes/'+str(i+1)+'.jpg', img)


def preprocess_augment(transformations=None, root='data/crop/', growth=10):
    image_files = [list(listdir_nohidden(os.path.join(root, file))) for file in ["no", "yes"]]
    where = 'no/'
    for file in image_files:
        counter = len(file) + 1
        for i in range(growth):
            for img in file:
                img_path = os.path.join(root, where, img)
                image = Image.open(img_path)
                if transformations:
                    image = transformations(image)
                image.save(root + where + str(counter) + '.jpg')
                counter += 1
        where = 'yes/'


def preprocess_aug(file, dataset, transformations=None, root='data/crop/', growth=10):
    counter = 1
    #print(len(dataset))
    for j in range(growth):
        for i in range(len(dataset)):
            image, label = dataset[i]
            if transformations:
                image = transformations(image)
            if label == 0:
                image.save(root + file + 'no/' + str(counter) + '.jpg')
            else:
                image.save(root + file + 'yes/' + str(counter) + '.jpg')
            counter += 1


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class TumorDataset(Dataset):
    def __init__(self, transformations=None, root='data/crop/'):
        self.root = root
        self.transformations = transformations
        self.image_files = [list(listdir_nohidden(os.path.join(self.root, file))) for file in ["no", "yes"]]

    def __len__(self):
        return sum([len(i) for i in self.image_files])
    
    def __getitem__(self, idx):
        where = "no"
        index = idx
        which = 0
        if idx >= len(self.image_files[0]):
            where = "yes"
            index -= len(self.image_files[0])
            which = 1
        img_path = os.path.join(self.root, where, self.image_files[which][index])
        image = Image.open(img_path)
        y_label = torch.tensor(which)
        if self.transformations:
            image = self.transformations(image)
        return (image, y_label)


