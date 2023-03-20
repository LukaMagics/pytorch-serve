import json
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

class PyTorchIrisHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
    
    def initialize(self, context): # context는 torchserve
        # Load the PyTorch model
        self._context = context
        print("------------- print context ---------- \n", context)
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get('model_dir')
        self.model = torch.load('/home/model-server/pytorch-serve/torch_iris.pth')
        self.model.eval()
        print("model loaded successfully")
        self.initialized = True
        
    def preprocess(self, data):
        # Parse the input data
        print("print original data var")
        data = json.loads(data[0]["data"].decode("utf-8"))
        print("-------- print data after parsing -------- \n", data)
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
