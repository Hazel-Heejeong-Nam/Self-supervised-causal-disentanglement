import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
       
        imgs = os.listdir(root)

        self.dataset = dataset
        
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        #print(self.imglabel)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        array1 = np.asarray(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)

def c_dataset(dataset_dir, batch_size, dataset="train"):
  
	dataset = dataload_withlabel(dataset_dir, dataset)
	dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

	return dataset