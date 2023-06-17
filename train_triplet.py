import torch
import torch.optim as optim
import os
from src.data import TripletFaceDataset 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from src.model.arcface import ft_model
# from loss_functions import AngularPenaltySMLoss
from src.utils.callbacks import SaveBestModel
from src.utils.misc import exponential_smoothing




torch.manual_seed(42)

# Implement functions here
train_losses =[]
def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
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
  
    for batch_idx, (anchor, positive, negative, _, _, _) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        out_anchor, out_positive, out_negative = model(anchor, positive, negative) # get the embeddings
        loss = criterion(out_anchor, out_positive, out_negative) # compute the loss
        loss.backward()
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

eval_losses=[]
def validate(model, device, test_loader, criterion, save_ckpt = True, callbacks = None ):
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
        for anchor, positive, negative, _, _, _ in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            out_anchor, out_positive, out_negative = model(anchor, positive, negative)
            loss = criterion(out_anchor, out_positive, out_negative)
            test_loss += loss.item()
        eval_losses.append(test_loss/len(test_loader))
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss/len(test_loader)))
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
    
    if save_ckpt:
        torch.save(model.backbone.state_dict(), "weight/model_finetune_triplet_1000.pth")

        if callbacks is not None:
            callbacks[0](test_loss/len(test_loader), model.backbone)
            
    
def train_triplet(model, train_path, test_path, epochs, batch_size = 32, device = None, roc_fig_path = "result/fig1.jpg"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    train_dataset = TripletFaceDataset(train_path)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers= 0, drop_last = True)
    test_dataset = TripletFaceDataset(test_path)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers= 0, drop_last = True)
    save_best_callback_model = SaveBestModel("weight/model_finetune_triplet_best_1000.pth")
    


    # define your model
    # model = FaceEmb(ft_model())
    # model = FaceEmb()
    model = model.to(device)
    # define the loss and optimizer
    def sim_loss(x1,x2):
        return 1 - F.cosine_similarity(x1,x2)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function = sim_loss, margin=0.5)
    # optimizer = optim.SGD(model.parameters(), lr=5e-5, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=5e-5,)

    # train and validate after each epoch (or specify it to be after a desired number)
   

    for epoch in range(1,epochs+1):
        train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)
        validate(model, device, testloader, criterion, callbacks= [save_best_callback_model])
    plt.plot(exponential_smoothing(train_losses, .2))
    plt.plot(exponential_smoothing(eval_losses, .2))
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig(roc_fig_path)
    plt.show()
