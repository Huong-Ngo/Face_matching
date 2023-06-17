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




def accuracy(root, weight_path: str,  threshold = 0.6):
  val_model = get_test_model(weight_path, 'weight/backbone_r100_glint360k.pt').cuda()
  dataset_test = TripletFaceDataset(root, 10)
  
  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                       shuffle=True, drop_last = True)
  
  cos = nn.CosineSimilarity(dim=0, eps=1e-6)
  gt =[]
  pr = []
  for a,p,n, _, _, _ in tqdm(test_loader):
    a_out,p_out,n_out = val_model(a.cuda(), p.cuda(),n.cuda())

    # a_out = get_embeded(a)
    # p_out = get_embeded(p)
    # n_out = get_embeded(n)
    # a_out , p_out = get_embeded_vecto(a[0],p[0])

    # a_out , n_out = get_embeded_vecto(a[0],n[0])

    # # print((a_out-n_out).mean())
    distance_ap = cos(a_out[0],p_out[0])
    # print(distance_ap)
    gt.append(1)
    pr.append(max(0,distance_ap.item()))
    distance_an = cos(a_out[0],n_out[0])
    # print(distance_an)
    gt.append(0)
    pr.append(max(0,distance_an.item()))
    pr_binary = []
    for i in range (len(pr)):
      if pr[i]>threshold:
        pr_binary.append(1)
      else:
        pr_binary.append(0)
  pr_binary = np.array(pr) > threshold

  c_m = cm(gt,pr_binary)
  precision = precision_score(gt,pr_binary)
  recall = recall_score(gt,pr_binary)
  F1_score = f1_score(gt,pr_binary)
  accuracy = accuracy_score(gt,pr_binary)

  
  print("precision:",precision)
  print("recall:",recall)
  print("F1 score:",F1_score)
  print("Accuracy:",accuracy)
  

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.matshow(c_m, cmap=plt.cm.Oranges, alpha=0.3)
  for i in range(c_m.shape[0]):
      for j in range(c_m.shape[1]):
          ax.text(x=j, y=i,s=c_m[i, j], va='center', ha='center', size='xx-large')
  
  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix', fontsize=18)
  plt.show()


  
  # Tính toán đường cong ROC
  fpr, tpr, thresholds = roc_curve(gt, pr)

  # Tính diện tích dưới đường cong ROC
  roc_auc = auc(fpr, tpr)

  # Vẽ đường cong ROC
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()

  # Tìm ngưỡng tối ưu theo đường cong ROC
  optimal_idx = np.argmax(tpr - fpr)
  optimal_threshold = thresholds[optimal_idx]
  pr_bi = []
  for i in range (len(pr)):
    if pr[i]>optimal_threshold:
      pr_bi.append(1)
    else:
      pr_bi.append(0)
  f1_best = f1_score(gt,pr_bi)
  print("Best threshold:",optimal_threshold)

  c_m_best = cm(gt,np.array(pr)>= optimal_threshold)
  precision_best = precision_score(gt,np.array(pr)>= optimal_threshold)
  recall_best = recall_score(gt,np.array(pr)>= optimal_threshold)
  F1_score_best = f1_score(gt,np.array(pr)>= optimal_threshold)
  accuracy_best = accuracy_score(gt,np.array(pr)>= optimal_threshold)

  print("precision:",precision_best)
  print("recall:",recall_best)
  print("F1 score:",F1_score_best)
  print("Accuracy:",accuracy_best)

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.matshow(c_m_best, cmap=plt.cm.Blues, alpha=0.3)
  for i in range(c_m_best.shape[0]):
      for j in range(c_m_best.shape[1]):
          ax.text(x=j, y=i,s=c_m_best[i, j], va='center', ha='center', size='xx-large')
  
  plt.xlabel('Predictions', fontsize=18)
  plt.ylabel('Actuals', fontsize=18)
  plt.title('Confusion Matrix', fontsize=18)
  plt.show()