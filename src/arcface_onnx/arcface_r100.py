import onnxruntime
import torch
import numpy as np

def get_session(weight_path):

    session = onnxruntime.InferenceSession(weight_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # outputs = session.run([model], inputs)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    session.get_inputs()[0]
    return session

def preprocess(image):
    """
    Tranpose dimension from (batch, width, height, channel) to (batch, channel, width, height)
    """
    image = np.transpose(image, (0, 3, 1, 2))
    return (image-127.5) / 128

def forward(image, session):
    input_name = session.get_inputs()[0].name
    image = preprocess(image).astype(np.float32)
    result = session.run(None, {input_name: image})
    return torch.from_numpy(result[0]).float()

