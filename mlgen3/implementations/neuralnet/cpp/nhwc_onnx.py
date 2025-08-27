import numpy as np
import onnx
import os
from collections import OrderedDict

from mlgen3.implementations.implementation import Implementation
from mlgen3.models.nn.activations import Sign, Sigmoid, Relu, Step
from mlgen3.models.nn.linear import Linear
from mlgen3.models.nn.batchnorm import BatchNorm

class NHWC_ONNX(Implementation):
    """
    ONNX-based implementation for neural networks.
    Instead of embedding weights and biases directly in the C++ code,
    this implementation generates code that loads them from an ONNX file.
    """

    def __init__(self, model, onnx_path, feature_type="float", label_type="float", internal_type="float", align=None):
        """
        Initialize NHWC_ONNX implementation.
        
        Args:
            model: The neural network model
            onnx_path: Path to the exported ONNX model
            feature_type: Type for input features in generated code
            label_type: Type for output labels in generated code
            internal_type: Type for internal calculations
            align: Memory alignment for optimization
        """
        super().__init__(model, feature_type, label_type)
        self.internal_type = internal_type
        self.align = align
        self.onnx_path = onnx_path
        
        # Verify that the ONNX file exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    def implement(self):
        """Generate C++ code that loads weights from the ONNX file."""
        # Load the ONNX model to get model structure info
        onnx_model = onnx.load(self.onnx_path)
        
        alloc = ""
        code = ""
        header = "#include <algorithm>\n#include <cmath>\n#include <fstream>\n#include <onnxruntime_c_api.h>\n"
        
        # Add code to dynamically load weights from ONNX
        code += self._generate_onnx_loader_code()
        
        # Add ONNX runtime initialization code
        code += self._generate_onnx_session_code()
        
        # Generate forward pass code based on the model structure
        code += self._generate_inference_code()
        
        # Combine the code
        self.code = f"""
            #include "model.h"
            {alloc}
            
            // Global ONNX session and environment
            static Ort::Env env;
            static Ort::Session* session = nullptr;
            static bool model_loaded = false;
            
            {code}
            
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
                if (!model_loaded) {{
                    initialize_onnx_model();
                }}
                
                return run_inference(x);
            }}
        """
        
        # Generate header file with necessary includes and function declarations
        self.header = f"""
            #pragma once
            #include <iostream>
            #include <ostream>
            #include <vector>
            #include <string>
            #include <onnxruntime_cxx_api.h>
            
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
            void initialize_onnx_model();
            std::vector<{self.label_type}> run_inference(std::vector<{self.feature_type}> &x);
        """.strip()
    
    def _generate_onnx_loader_code(self):
        """Generate code to load the ONNX model."""
        onnx_filename = os.path.basename(self.onnx_path)
        return f"""
            void initialize_onnx_model() {{
                try {{
                    // Initialize environment
                    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "MLGen3Model");
                    
                    // Session options
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(1);
                    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
                    
                    // Create session
                    session = new Ort::Session(env, "{onnx_filename}", session_options);
                    model_loaded = true;
                    
                    std::cout << "ONNX model loaded successfully." << std::endl;
                }}
                catch(const Ort::Exception& e) {{
                    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
                    model_loaded = false;
                }}
            }}
        """
    
    def _generate_onnx_session_code(self):
        """Generate code for managing the ONNX runtime session."""
        return f"""
            void cleanup_onnx_model() {{
                if (session) {{
                    delete session;
                    session = nullptr;
                    model_loaded = false;
                }}
            }}
        """
    
    def _generate_inference_code(self):
        """Generate code for model inference using ONNX Runtime."""
        input_shape = self.model.layers[0].input_shape
        output_shape = self.model.layers[-1].output_shape
        
        return f"""
            std::vector<{self.label_type}> run_inference(std::vector<{self.feature_type}> &x) {{
                // Create input tensor
                std::vector<float> input_tensor_values(x.begin(), x.end());
                std::vector<int64_t> input_shape = {{1, {input_shape}}};
                
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size()
                );
                
                // Define input and output names
                std::vector<std::string> input_names = session->GetInputNames();
                std::vector<std::string> output_names = session->GetOutputNames();
                
                // Convert to C-style strings for the Run method
                std::vector<const char*> input_names_char;
                std::vector<const char*> output_names_char;
                
                for (const auto& name : input_names) {{
                    input_names_char.push_back(name.c_str());
                }}
                
                for (const auto& name : output_names) {{
                    output_names_char.push_back(name.c_str());
                }}
                
                // Run inference
                auto output_tensors = session->Run(
                    Ort::RunOptions{{nullptr}}, 
                    input_names_char.data(), 
                    &input_tensor, 
                    1, 
                    output_names_char.data(), 
                    1
                );
                
                // Extract output data
                float* output_data = output_tensors[0].GetTensorMutableData<float>();
                std::vector<{self.label_type}> result({output_shape});
                
                for (int i = 0; i < {output_shape}; i++) {{
                    result[i] = static_cast<{self.label_type}>(output_data[i]);
                }}
                
                return result;
            }}
        """

