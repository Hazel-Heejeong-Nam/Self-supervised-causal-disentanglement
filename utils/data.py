import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import glob


class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root):
        self.imgs = glob.glob(root+'/*.png')
        self.imglabel = [list(map(int,imgpath.split('/')[-1].split('.')[0].split("_")[1:]))  for imgpath in self.imgs]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]
        
        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.array(pil_img)
        array1 = np.array(label)
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

def c_dataset(dataset_dir, batch_size, num_workers, shuffle:bool):
  
	dataset = dataload_withlabel(dataset_dir)
	loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return loader