import os
from mlgen3.implementations.implementation import Implementation
from mlgen3.implementations.neuralnet.cpp.onnx_utils import deploy_onnx_model

class VGG_ONNX(Implementation):
    """
    Implementation for VGG-style CNN models using ONNX Runtime.
    
    This implementation generates C++ code that uses ONNX Runtime to load and run
    VGG models with convolutional layers, max pooling, etc.
    
    Attributes:
        model: The MLGen3 model
        onnx_path: Path to the ONNX file
        feature_type: C++ type for input features
        label_type: C++ type for output labels
        internal_type: C++ type for internal computations
        batch_size: Batch size for inference
        input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        input_height: Height of input images
        input_width: Width of input images
    """
    
    def __init__(self, model, onnx_path, feature_type="float", label_type="int", internal_type="float", 
                 batch_size=128, input_channels=None, input_height=None, input_width=None):
        """
        Initialize the VGG_ONNX implementation.
        
        Args:
            model: MLGen3 model object
            onnx_path: Path to the ONNX model file
            feature_type: C++ type for input features
            label_type: C++ type for output labels
            internal_type: C++ type for internal computations
            batch_size: Batch size for inference
            input_channels: Number of input channels (default: auto-detect from model)
            input_height: Height of input images (default: auto-detect from model)
            input_width: Width of input images (default: auto-detect from model)
        """
        super().__init__(model, feature_type, label_type)
        self.onnx_path = onnx_path
        self.internal_type = internal_type
        self.batch_size = batch_size
        self.model.onnx_path = onnx_path  # Store path for later use by materializer
        
        # Determine input dimensions from model data if not specified
        if input_channels is None or input_height is None or input_width is None:
            # Try to infer from XTest if available (assumed to be in NCHW format)
            if hasattr(model, 'XTest') and model.XTest is not None and len(model.XTest.shape) == 4:
                _, channels, height, width = model.XTest.shape
                self.input_channels = input_channels if input_channels is not None else channels
                self.input_height = input_height if input_height is not None else height
                self.input_width = input_width if input_width is not None else width
            else:
                # Default to MNIST/Fashion-MNIST dimensions if no data available
                self.input_channels = input_channels if input_channels is not None else 1
                self.input_height = input_height if input_height is not None else 28
                self.input_width = input_width if input_width is not None else 28
        else:
            self.input_channels = input_channels
            self.input_height = input_height
            self.input_width = input_width
        
    def implement(self):
        """Generate C++ code for VGG ONNX model inference."""
        
        # ONNX Runtime headers
        onnx_headers = """
        #include <onnxruntime_cxx_api.h>
        #include <array>
        #include <iostream>
        #include <vector>
        #include <cassert>
        """
        
        # Model implementation that loads and runs the ONNX model
        implementation = f"""
        #include "model.h"
            
            
            // Global ONNX session and environment
            static Ort::Env env;
            static Ort::Session* session = nullptr;
            static bool model_loaded = false;
            
            // Input dimensions
            static const int INPUT_CHANNELS = {self.input_channels};
            static const int INPUT_HEIGHT = {self.input_height};
            static const int INPUT_WIDTH = {self.input_width};
            
            
            void initialize_onnx_model() {{
                try {{
                    // Initialize environment
                    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "MLGen3Model");
                    
                    // Session options
                    Ort::SessionOptions session_options;
                    session_options.SetIntraOpNumThreads(1);
                    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
                    
                    // Create session
                    session = new Ort::Session(env, "{os.path.basename(self.onnx_path)}", session_options);
                    model_loaded = true;
                    
                    std::cout << "ONNX model loaded successfully." << std::endl;
                }}
                catch(const Ort::Exception& e) {{
                    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
                    model_loaded = false;
                }}
            }}
        
            void cleanup_onnx_model() {{
                if (session) {{
                    delete session;
                    session = nullptr;
                    model_loaded = false;
                }}
            }}
        
            std::vector<{self.label_type}> run_inference(std::vector<{self.feature_type}> &x) {{
                // Calculate the actual batch size from the input data
                // For CNN input with shape [batch_size, channels, height, width]
                int input_size = x.size();
                int sample_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
                int actual_batch_size = input_size / sample_size;
                
                if (actual_batch_size * sample_size != input_size) {{
                    std::cerr << "Input size " << input_size << " is not divisible by sample size " 
                              << sample_size << " (" << INPUT_CHANNELS << "*" << INPUT_HEIGHT << "*" << INPUT_WIDTH << ")." << std::endl;
                    return std::vector<{self.label_type}>();
                }}
                
                // Create input tensor with the correct batch size
                std::vector<float> input_tensor_values(x.begin(), x.end());
                std::vector<int64_t> input_shape = {{actual_batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH}};
                
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
                std::vector<{self.label_type}> result({self.model.layers[-1].output_shape});
                
                // For simplicity, we're only returning the results for the first sample in the batch
                // In a real application, you might want to return results for all samples
                for (int i = 0; i < {self.model.layers[-1].output_shape}; i++) {{
                    result[i] = static_cast<{self.label_type}>(output_data[i]);
                }}
                
                return result;
            }}
        
            
            std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
                if (!model_loaded) {{
                    initialize_onnx_model();
                }}
                
                return run_inference(x);
            }}
        """
        
        # Header file with function declarations
        header = f"""
        #pragma once
        
        {onnx_headers}
        
        std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x);
        void initialize_onnx_model();
        void cleanup_onnx_model();
        """
        
        self.code = implementation
        self.header = header.strip()
