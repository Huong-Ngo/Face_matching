from src.data import *
import torch
from tqdm import tqdm
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt 
import torch.nn as nn
from src.utils.model import get_triplet_model, get_test_model


def process_data(in_data_path: str, out_data_path: str, weight_path):
    model = get_test_model(weight_path, init_weight='weight/backbone_r100_glint360k.pt')
    model = model.cuda()
    model.eval()
    list_cs = []
    count = 0
    with torch.no_grad(): 
        dataset_test = Triplet_loader(in_data_path)
        for idx, (a,p,n,_,_,_) in tqdm(enumerate(dataset_test)):

            a,p,n = a.unsqueeze(0).cuda(), p.unsqueeze(0).cuda(), n.unsqueeze(0).cuda()

            ea, ep, en = model(a,p,n)
            cs = nn.CosineSimilarity()
            dis_ap = cs(ea,ep)[0].item()
            if dis_ap > 0.3:
                count += 1
                with open(in_data_path, "r") as input:
                    lines = input.readlines()
                with open(out_data_path, "a") as output:
                        output.write(lines[idx])
                                        

                # img: torch.Tensor = torch.cat([a.squeeze(0).cpu(), p.squeeze(0).cpu(), n.squeeze(0).cpu()], axis=-1)
                # img = (img - img.min()) / (img.max() - img.min())
                # img = (TF.to_pil_image(img))
                
                # plt.imshow(img)
                # plt.show()

            dis_an = cs(ea,en)[0].item()
            dis_pn = cs(ep,en)[0].item()
            list_cs.append((dis_ap, dis_an, dis_pn))
            


    # print(count)
