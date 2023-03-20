import torch
import torch.nn.functional as F
import json

class PyTorchIrisHandler:
    def __init__(self):
        # Load the PyTorch model
        self.model = torch.load('/home/model-server/pytorch-serve/torch_iris.pth')
        self.model.eval()
        print("model loaded successfully")
        
    def preprocess(self, data):
        # Parse the input data
        input_data = torch.tensor(data).float()
        return input_data
    
    def inference(self, input_data):
        # Make predictions
        with torch.no_grad():
            output = self.model(input_data)
        return output
    
    def postprocess(self, inference_output):
        # Convert PyTorch output tensor to a NumPy array
        output = inference_output.numpy().tolist()
        return {'output': output}
