from src.utils.model import get_test_model
import torch
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from src.data import TripletFaceDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from torchmetrics.classification.accuracy import BinaryAccuracy



@torch.no_grad()
def validation(val_model:torch.nn.Module,  test_loader, criterion):
    device = next(val_model.parameters()).device
    val_model.eval()
    test_loss = 0

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    gt =[]
    pr = []
    for a,p,n, _, _, _ in tqdm(test_loader):
        a_out,p_out,n_out = val_model(a.to(device), p.to(device),n.to(device))  #[N,512]

        loss = criterion(a_out,p_out,n_out)
        test_loss += loss.item()

        distance_ap = torch.clamp(cos(a_out,p_out), 0, 1).cpu() #[N,]
        distance_an = torch.clamp(cos(a_out,n_out), 0, 1).cpu()

        distance_cat = torch.cat([distance_ap, distance_an], dim = 0)  # [2*N,]
        # print(distance_cat.device)
        target_cat = torch.cat([torch.ones_like(distance_ap), torch.zeros_like(distance_an)], dim = 0) # [2*N,]
        # print(target_cat.device)
        pr.append(distance_cat) #[num_batches, 2*N,]
        gt.append(target_cat)

    pr = torch.cat(pr, dim= 0).tolist() # [num_batches*2*N, ]
    gt = torch.cat(gt, dim= 0).tolist()
    
    test_loss= test_loss/ len(test_loader)
    
    # Tính toán đường cong ROC
    fpr, tpr, thresholds = roc_curve(gt, pr)

    # Tìm ngưỡng tối ưu theo đường cong ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print("Best threshold:",optimal_threshold)


    precision_best = precision_score(gt,np.array(pr)>= optimal_threshold)
    recall_best = recall_score(gt,np.array(pr)>= optimal_threshold)
    F1_score_best = f1_score(gt,np.array(pr)>= optimal_threshold)
    accuracy_best = accuracy_score(gt,np.array(pr)>= optimal_threshold)

    print("precision:",precision_best)
    print("recall:",recall_best)
    print("F1 score:",F1_score_best)
    print("Accuracy:",accuracy_best)

    return F1_score_best, test_loss

 


