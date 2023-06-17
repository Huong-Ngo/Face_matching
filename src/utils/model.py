import torch
from ..model.arcface import iresnet100, Iresnet_finetune, FaceEmb

def get_triplet_model(weight_path: str= None):
    if weight_path is not None:
        if weight_path.split('.')[-1] == 'pt':
            arcface_model = torch.load(weight_path, map_location='cpu')
        else:
            arcface_model = iresnet100()
            arcface_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        
        # for name, param in arcface_model.named_parameters():
        #     param.requires_grad = False
        
    else:
        arcface_model = iresnet100()

    ft_model = Iresnet_finetune(arcface_model)
    model = FaceEmb(ft_model)
    return model


def get_test_model(weight_path: str = None, init_weight = None):
    triplet_model = get_triplet_model(init_weight)
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location='cpu')
        if 'model_state_dict' in state_dict.keys():
            state_dict = state_dict['model_state_dict']
        # triplet_model.backbone.load_state_dict(state_dict)
        triplet_model.backbone.load_state_dict(state_dict)

    triplet_model.eval()
    return triplet_model


def freeze_layer(model: torch.nn.Module, layer_name: list ):
    for name, param in model.named_parameters():
        for key in layer_name:
            if key not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                break

