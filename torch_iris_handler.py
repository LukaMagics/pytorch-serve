import os
import json
import numpy
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from torch_iris import PytorchIrisModel

class PyTorchIrisHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
    
    # initialize, preprocess 등 BaseHandler의 메소드 명과 동일한 경우 상속 후 override되어 부모클래스의 메소드는 무시되어 재정의됨
    def initialize(self, context): # context는 torchserve에서 자동할당해줌.
        # Load the PyTorch model
        self._context = context
        print("------------- print context ---------- \n", context)
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get('model_dir')  # model_dir == --model-store 경로
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        self.model = PytorchIrisModel()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.eval()
        print("model loaded successfully")
        self.initialized = True
        
    def preprocess(self, data):  # data는 torchserve에서 자동할당해줌. data (list): List of the data from the request input.
        # Parse the input data
        print("print original data var", data)
        #data = json.loads(data[0].get("body"))
        preprocessed_data = json.loads(data[0].get("instances"))
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        print("-------- print data after parsing -------- \n", data)
        preprocessed_data = torch.tensor(data).float()
        return preprocessed_data  # return은 torch의 tensor형으로 변형해서
    
    def inference(self, model_input):
        print("inference start")
        # Make predictions
        with torch.no_grad():
            model_output = self.model(model_input)
        return model_output 
    
    def postprocess(self, inference_output):
        # Convert PyTorch output tensor to a NumPy array
        postprocess_output = inference_output.numpy().tolist()
        return postprocess_output
    
    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)