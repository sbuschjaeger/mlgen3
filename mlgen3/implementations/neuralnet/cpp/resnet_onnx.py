import os
import numpy as np
import onnx
import onnxruntime as ort

from mlgen3.implementations.implementation import Implementation
from mlgen3.implementations.neuralnet.cpp.onnx_utils import deploy_onnx_model

class ResNetONNX(Implementation):
    """ResNet implementation using ONNX runtime for inference.
    
    This implementation generates C++ code that uses the ONNX runtime API
    to perform inference with a ResNet model exported to ONNX format.
    It supports the characteristic residual (skip) connections of ResNet models.
    """
    
    def __init__(self, model, onnx_path=None, feature_type="float", label_type="int",  internal_type="float",
                 batch_size=128, input_channels=None, input_height=None, input_width=None):
        """Initialize the ResNet ONNX implementation.
        
        Args:
            model: The MLGen3 model (can be None if onnx_path is provided)
            feature_type: Data type for inputs (default: "float")
            label_type: Data type for outputs (default: "float")
            onnx_path: Path to the ONNX model file (if model is None)
        """
        super().__init__(model, feature_type, label_type)
        
        if onnx_path is not None:
            self.onnx_path = onnx_path
        elif hasattr(model, 'onnx_path'):
            self.onnx_path = model.onnx_path
        else:
            raise ValueError("Either model with onnx_path attribute or onnx_path parameter must be provided")
        
        self.internal_type = internal_type
        self.batch_size = batch_size
        # Load ONNX model to extract metadata
        self.onnx_model = onnx.load(self.onnx_path)
        self.session = ort.InferenceSession(self.onnx_path)
        
        # Extract input shape information
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape and determine if normalization is needed
        input_shape = self.session.get_inputs()[0].shape
        
        # Handle dynamic batch dimensions (often set as None or -1 in exported models)
        self.batch_size = 1  # Default batch size
        if input_shape[0] is None or input_shape[0] == -1:
            input_shape = list(input_shape)
            input_shape[0] = self.batch_size
        
        # Extract dimensions (handle different formats: NCHW or NHWC)
        if len(input_shape) == 4:  # Image input
            # Assuming NCHW format (most common in ONNX)
            self.batch_size = input_shape[0]
            self.channels = input_shape[1]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            self.input_size = self.channels * self.input_height * self.input_width
        else:  # Handle flat inputs
            self.input_size = input_shape[1]
            self.channels = 1
            self.input_height = 1
            self.input_width = self.input_size
        
        # Get output shape
        output_shape = self.session.get_outputs()[0].shape
        if len(output_shape) == 2:
            self.num_classes = output_shape[1]
        else:
            self.num_classes = output_shape[0]

    def implement(self):
        """Generate C++ code for the ResNet ONNX model."""
        # Common header includes
        header = """
            #pragma once
            #include <vector>
            #include <string>
            #include <onnxruntime_cxx_api.h>
            
            std::vector<{0}> predict(std::vector<{1}> &x);
        """.format(self.label_type, self.feature_type).strip()
        
        # Implementation for model loading and inference
        code = """
            #include "model.h"
            #include <algorithm>
            #include <numeric>
            #include <cassert>
            #include <cmath>
            #include <iostream>
            #include <vector>
            #include <memory>
            
            // ONNX Runtime inference code for ResNet model
            std::vector<{0}> predict(std::vector<{1}> &x) {{
                // Verify input size
                if (x.size() != {2}) {{
                    std::cerr << "Error: Expected input size " << {2} << " but got " << x.size() << std::endl;
                    return std::vector<{0}>();
                }}
                
                // Create environment
                Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNetONNXModel");
                
                // Session options
                Ort::SessionOptions session_options;
                session_options.SetIntraOpNumThreads(1);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                
                // Load model
                const char* model_path = "{3}";
                Ort::Session session(env, model_path, session_options);
                
                // Input/output names
                Ort::AllocatorWithDefaultOptions allocator;
                const char* input_name = "{4}";
                const char* output_name = "{5}";
                
                // Create input tensor
                std::vector<float> input_tensor_values(x.begin(), x.end());
                
                // Define fixed batch size for inference
                const int64_t inference_batch_size = 1;
                std::vector<int64_t> input_shape = {{inference_batch_size, {7}, {8}, {9}}};
                
                // Memory for input and output tensors
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size());
                
                // Define output names
                std::vector<const char*> input_names = {{input_name}};
                std::vector<const char*> output_names = {{output_name}};
                
                // Run inference
                auto output_tensors = session.Run(
                    Ort::RunOptions{{nullptr}}, 
                    input_names.data(), &input_tensor, 1, 
                    output_names.data(), 1
                );
                
                // Get output tensor
                Ort::Value& output_tensor = output_tensors.front();
                
                // Convert to vector
                auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
                int64_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
                
                // Copy output to result vector
                std::vector<{0}> result(output_size);
                const float* output_data = output_tensor.GetTensorData<float>();
                for (int64_t i = 0; i < output_size; ++i) {{
                    result[i] = static_cast<{0}>(output_data[i]);
                }}
                
                return result;
            }}
        """.format(
            self.label_type,
            self.feature_type,
            self.input_size,
            os.path.basename(self.onnx_path),  # Just the filename, not the full path
            self.input_name,
            self.output_name,
            self.batch_size,
            self.channels,
            self.input_height,
            self.input_width
        ).strip()
        
        self.header = header
        self.code = code
        
        return {
            "Sources": {
                "model.h": self.header,
                "model.cpp": self.code
            }
        }

