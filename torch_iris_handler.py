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
        print("initialize start")
        self._context = context
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
        print("initialize end")
        self.initialized = True
        
    def preprocess(self, data):  # data는 torchserve에서 자동할당해줌. data (list): List of the data from the request input.
        print("preprocess start")
        # Parse the input data
        #data = json.loads(data[0].get("body"))
        # Extract the JSON string from the bytearray
        json_str = data[0]['body'].decode('utf-8')

        # Parse the JSON string into a Python dictionary
        data = json.loads(json_str)

        # Access the list of instances
        instances = data['instances']
        preprocessed_data = torch.tensor(instances).float()
        print("preprocess end")

        return preprocessed_data  # return은 torch의 tensor형으로 변형해서
    
    def inference(self, preprocessed_data):
        print("inference start")
        # Make predictions
        inference_output = []
        with torch.no_grad():
            model_input = preprocessed_data
            model_output = self.model(model_input)
            inference_output = model_output.argmax(-1).tolist()
        print("inference end")

        return inference_output
    
    def postprocess(self, inference_output):
        print("postprocess start")
        # Convert PyTorch output tensor to a NumPy array
        iris_dict = {0 : 'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}
        labeled_output = [iris_dict[o] for o in inference_output]
        postprocess_output = {
            "output": labeled_output
        }
        postprocess_output = [postprocess_output]
        print("postprocess_output : ", postprocess_output)
        print("postprocess end")
        return postprocess_output
    
    def handle(self, data, context):
        if not self.initialized:
          self.initialized(context)
        
        preprocessed_data = self.preprocess(data)
        inference_output = self.inference(preprocessed_data)
        final_output = self.postprocess(inference_output)
        return final_output