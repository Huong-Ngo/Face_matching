from onnx2torch import convert
import torch

torch_model = convert(r'weight/arcfaces_resnet100.onnx')
torch.save(torch_model, r'weight/arcfaces_resnet100.pth')
torch_model.state_dict()