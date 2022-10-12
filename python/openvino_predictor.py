import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from openvino.runtime import Core

class Predictor:
    def __init__(self, model_path):
        ie_core = Core()
        model = ie_core.read_model(model=model_path)
        self.compiled_model = ie_core.compile_model(model=model, device_name="CPU")
        
    def get_inputs_name(self, num):
        return self.compiled_model.input(num)
    
    def get_outputs_name(self, num):
        return self.compiled_model.output(num)
    
    def predict(self, input_data):
        return self.compiled_model([input_data])
        