
import torch
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self,weight_path, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.weight_path = weight_path
        
    def __call__(self, current_valid_loss, 
                model, epoch = None, optimizer = None, criterion = None):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}\n")
            if epoch is not None:
                print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'loss': current_valid_loss,
                'criterion': criterion,
                }, self.weight_path)