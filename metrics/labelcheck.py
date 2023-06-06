import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")



def check_label(representation_function):
    
    eval_list = sorted(glob.glob('/mnt/hazel/data/causal_data/pendulum/eval/*.png'))
    assert len(eval_list) ==8
    print('          |   i    | i_label |   j   | j_label |  shade | s_label |   mid   | m_label |')
    print('=======================================================================================')
    
    
    for imgpath in eval_list :
        img = Image.open(imgpath)
        img = ToTensor()(img).to(device)
        i,j,shade,mid = imgpath.split('/')[-1].split('.')[0].split('_')[1:]
        label = representation_function(img)[0]
        print(f'              {i}       {j}       {shade}      {mid}      {label[0]:.3f}    {label[1]:.3f}     {label[2]:.3f}    {label[3]:.3f}')
        
if __name__ =='__main__':
    pass