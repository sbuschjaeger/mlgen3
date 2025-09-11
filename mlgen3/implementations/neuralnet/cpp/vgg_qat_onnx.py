import os
from mlgen3.implementations.implementation import Implementation
from mlgen3.implementations.neuralnet.cpp.onnx_utils import deploy_onnx_model

class VGG_QAT_ONNX(Implementation):
    """
    Implementation for Quantized VGG-style CNN models using ONNX Runtime.
    
    This implementation generates C++ code that uses ONNX Runtime to load and run
    quantized VGG models with specified bit-width (8, 4, or 2 bits).
    
    Attributes:
        model: The MLGen3 model
        onnx_path: Path to the ONNX file
        feature_type: C++ type for input features
        label_type: C++ type for output labels
        internal_type: C++ type for internal computations
        batch_size: Batch size for inference
        bit_width: Quantization bit width (8, 4, or 2)
        input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        input_height: Height of input images
        input_width: Width of input images
    """
    
    def __init__(self, model, onnx_path, feature_type="float", label_type="int", internal_type="float", 
                 batch_size=128, bit_width=8, input_channels=None, input_height=None, input_width=None):
        """
        Initialize the VGG_QAT_ONNX implementation.
        
        Args:
            model: MLGen3 model object
            onnx_path: Path to the ONNX model file
            feature_type: C++ type for input features
            label_type: C++ type for output labels
            internal_type: C++ type for internal computations
            batch_size: Batch size for inference
            bit_width: Quantization bit width (8, 4, or 2)
            input_channels: Number of input channels (default: auto-detect from model)
            input_height: Height of input images (default: auto-detect from model)
            input_width: Width of input images (default: auto-detect from model)
        """
        super().__init__(model, feature_type, label_type)
        self.onnx_path = onnx_path
        self.internal_type = internal_type
        self.batch_size = batch_size
        self.bit_width = bit_width
        self.model.onnx_path = onnx_path
        
        # Validate bit_width
        if bit_width not in [8, 4, 2]:
            raise ValueError(f"Bit width must be 8, 4, or 2. Received: {bit_width}")
        
        # Determine input dimensions from model data if not specified
        if input_channels is None or input_height is None or input_width is None:
            # Try to infer from XTest if available (assumed to be in NCHW format)
            if hasattr(model, 'XTest') and model.XTest is not None and len(model.XTest.shape) == 4:
                _, channels, height, width = model.XTest.shape
                self.input_channels = input_channels if input_channels is not None else channels
                self.input_height = input_height if input_height is not None else height
                self.input_width = input_width if input_width is not None else width
            else:
                raise ValueError("Input dimensions (channels, height, width) must be specified if XTest is not available.")
        else:
            self.input_channels = input_channels
            self.input_height = input_height
            self.input_width = input_width
        
    def implement(self):
        """Generate C++ code for Quantized VGG ONNX model inference."""
        
        # ONNX Runtime headers
        onnx_headers = """
        #include <onnxruntime_cxx_api.h>
        #include <array>
        #include <iostream>
        #include <vector>
        #include <cassert>
        #include <memory>
        #include <algorithm>
        """
        
        # Model implementation that loads and runs the quantized ONNX model
        implementation = f"""
        #include "model.h"
        
        // Global ONNX session and environment
        static Ort::Env env;
        static std::unique_ptr<Ort::Session> session;
        static bool model_loaded = false;
        
        // Input dimensions
        static const int INPUT_CHANNELS = {self.input_channels};
        static const int INPUT_HEIGHT = {self.input_height};
        static const int INPUT_WIDTH = {self.input_width};
        
        // Quantization parameters
        static const int QUANT_BIT_WIDTH = {self.bit_width};
        
        void initialize_onnx_model() {{
            try {{
                // Initialize environment
                env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "MLGen3QuantModel");
                
                // Session options
                Ort::SessionOptions session_options;
                session_options.SetIntraOpNumThreads(1);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                
                // Create session
                session = std::make_unique<Ort::Session>(env, "{os.path.basename(self.onnx_path)}", session_options);
                model_loaded = true;
                
                std::cout << "Quantized ONNX model ({self.bit_width}-bit) loaded successfully." << std::endl;
                
                // Print model input/output info
                Ort::AllocatorWithDefaultOptions allocator;
                auto input_name = session->GetInputNameAllocated(0, allocator);
                auto output_name = session->GetOutputNameAllocated(0, allocator);
                
                auto input_type_info = session->GetInputTypeInfo(0);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                auto input_dims = input_tensor_info.GetShape();
                
                auto output_type_info = session->GetOutputTypeInfo(0);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_dims = output_tensor_info.GetShape();
                
                std::cout << "Model input: " << input_name.get() << ", shape: [";
                for (const auto& dim : input_dims) {{
                    std::cout << dim << ", ";
                }}
                std::cout << "]" << std::endl;
                
                std::cout << "Model output: " << output_name.get() << ", shape: [";
                for (const auto& dim : output_dims) {{
                    std::cout << dim << ", ";
                }}
                std::cout << "]" << std::endl;
            }}
            catch(const Ort::Exception& e) {{
                std::cerr << "ONNX Runtime error during initialization: " << e.what() << std::endl;
                model_loaded = false;
            }}
            catch(const std::exception& e) {{
                std::cerr << "Standard error during initialization: " << e.what() << std::endl;
                model_loaded = false;
            }}
        }}
        
        void cleanup_onnx_model() {{
            if (session) {{
                session.reset();
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
            
            try {{
                // Create input tensor with the correct batch size
                std::vector<float> input_tensor_values(x.begin(), x.end());
                std::vector<int64_t> input_shape = {{actual_batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH}};
                
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_tensor_values.data(), input_tensor_values.size(),
                    input_shape.data(), input_shape.size()
                );
                
                // Define input and output names
                Ort::AllocatorWithDefaultOptions allocator;
                auto input_name = session->GetInputNameAllocated(0, allocator);
                auto output_name = session->GetOutputNameAllocated(0, allocator);
                
                const std::array<const char*, 1> input_names = {{input_name.get()}};
                const std::array<const char*, 1> output_names = {{output_name.get()}};
                
                // Run inference
                auto output_tensors = session->Run(
                    Ort::RunOptions{{nullptr}}, 
                    input_names.data(), 
                    &input_tensor, 
                    1, 
                    output_names.data(), 
                    1
                );
                
                // Get output shape info
                auto output_type_info = session->GetOutputTypeInfo(0);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_dims = output_tensor_info.GetShape();
                
                // Calculate number of output classes
                int num_classes = 1;
                if (output_dims.size() >= 2) {{
                    num_classes = output_dims[1];  // Shape is [batch_size, classes]
                }}
                
                // Extract output data
                float* output_data = output_tensors[0].GetTensorMutableData<float>();
                std::vector<{self.label_type}> result(num_classes);
                
                // Copy output data for the first sample
                for (int i = 0; i < num_classes; i++) {{
                    result[i] = static_cast<{self.label_type}>(output_data[i]);
                }}
                
                return result;
            }}
            catch(const Ort::Exception& e) {{
                std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
                return std::vector<{self.label_type}>();
            }}
            catch(const std::exception& e) {{
                std::cerr << "Standard error during inference: " << e.what() << std::endl;
                return std::vector<{self.label_type}>();
            }}
        }}
        
        std::vector<{self.label_type}> predict(std::vector<{self.feature_type}> &x) {{
            if (!model_loaded) {{
                initialize_onnx_model();
                if (!model_loaded) {{
                    std::cerr << "Failed to load ONNX model, cannot run inference" << std::endl;
                    return std::vector<{self.label_type}>();
                }}
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
