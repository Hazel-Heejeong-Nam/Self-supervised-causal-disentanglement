import glob
from PIL import Image
from torchvision.transforms import ToTensor
import torch
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



def check_label(representation_function):
    softmax = torch.nn.Softmax(dim=0)
    ce_cri = torch.nn.CrossEntropyLoss()

    eval_list = sorted(glob.glob('/home/work/YAI-Summer/hazel/data/causal_data/pendulum/eval/*.png'))
    assert len(eval_list) ==8
    label_dict = {}
    for imgpath in eval_list :
        img = Image.open(imgpath)
        img = ToTensor()(img).to(device)
        latent = imgpath.split('/')[-1].split('.')[0].split('_')[0]
        label = representation_function(img)[0]
        if latent not in label_dict.keys():
            label_dict[latent] = [label]
        else :
            label_dict[latent].append(label)

    loss_list = []
    target_list = []
    key_list = []
    for key in label_dict.keys():
        assert len(label_dict[key])==2
        #score = softmax(torch.abs(label_dict[key][0] - label_dict[key][1]))
        score = torch.abs(label_dict[key][0] - label_dict[key][1])
        print(score)
        target = torch.argmax(softmax(score))
        loss=ce_cri(score, target)
        print(f'factor : {key}       target : {target}       loss : {loss:.4f} \n')
        loss_list.append(loss)
        target_list.append(target)
        key_list.append(key)
    
    return sum(loss_list)/len(loss_list), target_list,key_list
    
if __name__ =='__main__':
    pass