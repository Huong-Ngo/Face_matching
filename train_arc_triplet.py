import torch
import torch.optim as optim
import os
from src.data import TripletFaceDataset 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
# from loss_functions import AngularPenaltySMLoss
from src.utils.callbacks import SaveBestModel, ModelCheckpoint
from test import validation
from src.utils.misc import exponential_smoothing
from src.utils.loss import arcface_triplet_loss

from torch.utils.tensorboard import SummaryWriter


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()





torch.manual_seed(42)

# Implement functions here
train_losses =[]
def train_one_epoch(model, arcface, device, train_loader, optimizer, criterion, epoch):
    """
    A method to train the model for one epoch

    Parameters:
            model: your created model
            device: specify the GPU or CPU
            train_loader: dataloader for training set
            optimizer: the traiining optimizer
            criterion: the loss function
            epoch: current epoch (int)
    """

    

    log_interval = 1 # specify to show logs every how many iterations 
    model.train()
    running_loss = 0
    train_loss = 0

    for batch_idx, (anchor, positive, negative, a_label, p_label, n_label) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        a_label, p_label, n_label = a_label.to(device), p_label.to(device), n_label.to(device)

        optimizer.zero_grad()
        feature_anchor, feature_positive, feature_negative = model(anchor, positive, negative) # get the embeddings
        out_anchor, out_positive, out_negative = arcface(feature_anchor, a_label), arcface(feature_positive, p_label), arcface(feature_negative, n_label)
        loss = criterion(out_anchor,out_positive,out_negative,a_label,p_label,n_label) # compute the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        train_loss += loss.item()
   
        
        # print the logs
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss/log_interval))
            running_loss = 0
    train_losses.append(train_loss/len(train_loader))
    return train_loss/len(train_loader)

eval_losses=[]
train_val_loss=[]
def validate(model, arcface, device, test_loader, criterion, check_train = False, save_ckpt = True, callbacks = None):
    """
    a method to validate the model

    Parameters:
            model: your created model
            device: specify to use GPU or CPU
            test_loader: The dataloader for testing
            criterion: the loss function
    
    """
    torch.manual_seed(100)
    model.eval()
    test_loss = 0
    iters = 0
    with torch.no_grad():
        for anchor, positive, negative, a_label, p_label, n_label in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            feature_anchor, feature_positive, feature_negative = model(anchor, positive, negative) # get the embeddings
            loss = criterion(feature_anchor, feature_positive, feature_negative)
            test_loss += loss.item()
        if not check_train:
            eval_losses.append(test_loss/len(test_loader))
        else:
            train_val_loss.append(test_loss/len(test_loader))
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss/len(test_loader)))
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')

    




def train_arc_triplet(model, arcface, train_path, test_path, epochs, batch_size = 32, device = None, roc_fig_path = "result/fig1.jpg"):
    writer = SummaryWriter(f'runs/arcface_s={str(arcface.s)}_m={str(arcface.m)}')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    train_dataset = TripletFaceDataset(train_path)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers= 0, drop_last = True)
    test_dataset = TripletFaceDataset(test_path)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(.75 * batch_size),
                                        shuffle=True, num_workers= 0, drop_last = True)


    callback_acc = ModelCheckpoint(root_dir='weight_triplet_arc',criterion_name= 'val_F1',mode= 'max', top_k= 1,save_last= True)

    model = model.to(device)
    arcface = arcface.to(device)
    # define the loss and optimizer
    def sim_loss(x1,x2):
        return 1 - F.cosine_similarity(x1,x2)
    test_criterion = nn.TripletMarginWithDistanceLoss(distance_function = sim_loss, margin=1.0)
    # criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = arcface_triplet_loss
    optimizer = optim.Adam(
        list(model.parameters()) + list(arcface.parameters()),
          lr=5e-5,)
    
    validation(model,testloader, test_criterion)

    # train and validate after each epoch (or specify it to be after a desired number)
    for epoch in range(1,epochs+1):
        train_loss = train_one_epoch(model, arcface, device, trainloader, optimizer, criterion, epoch)
        
        # validate(model, device, trainloader, test_criterion, True, False)
        F1_score, loss = validation(model,testloader, test_criterion)
        writer.add_scalar('validation_F1_score',
                                F1_score,epoch)
        writer.add_scalar('validation_loss',
                                loss,epoch)
        writer.add_scalar('training_loss',
                                train_loss,epoch)
        # validate(model, arcface, device, testloader, test_criterion)

        

        if callback_acc is not None:
            callback_acc(model,F1_score, epoch)
   

