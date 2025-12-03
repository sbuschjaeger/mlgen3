def check_config(config):
    """
    Check if the configuration file is valid.
    
    Args:
        config (dict): Configuration dictionary    
    Raises:
        ValueError: If the configuration file is not valid.
    """

    if config is None:
        raise ValueError("Configuration file is None. Please provide a valid configuration file.")
    
    if not isinstance(config, dict):
        raise ValueError("Configuration file must be a dictionary.")
    
    # Check if all required keys are present
    required_keys = ['quantization', 'model', 'training', 'evaluation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in configuration file.")


    ### Check quantization settings
    if 'quantization' in config:
        quantization = config['quantization']
        
        if 'use_matquant' not in quantization:
            raise ValueError("Missing 'use_matquant' key in quantization settings.")
        if not isinstance(quantization['use_matquant'], bool):
            raise ValueError("'use_matquant' must be a boolean value.")
        
        if 'target_bits' not in quantization:
            raise ValueError("Missing 'target_bits' key in quantization settings (use [32] for full-precision).")
        if not isinstance(quantization['target_bits'], list):
            raise ValueError("'target_bits' must be a list of positive integers.")
        for bit in quantization['target_bits']:
            if not isinstance(bit, int) or bit <= 0:
                raise ValueError("All elements in 'target_bits' must be positive integers.")
            
        if 'quantize_input' not in quantization:
            quantization['quantize_input'] = True  # Default to quantizing inputs
        else:
            if not isinstance(quantization['quantize_input'], bool):
                raise ValueError("'quantize_input' must be a boolean value.")

        if 'quantize_weight' not in quantization:
            quantization['quantize_weight'] = True  # Default to quantizing weights
        else:
            if not isinstance(quantization['quantize_weight'], bool):
                raise ValueError("'quantize_weight' must be a boolean value.")
            
        if 'quantize_activation' not in quantization:
            quantization['quantize_activation'] = True  # Default to quantizing activations
        else:
            if not isinstance(quantization['quantize_activation'], bool):
                raise ValueError("'quantize_activation' must be a boolean value.")
        
        # Check bias quantization flag (add default if missing)
        if 'quantize_bias' not in quantization:
            quantization['quantize_bias'] = True  # Default to quantizing bias
        else:
            if not isinstance(quantization['quantize_bias'], bool):
                raise ValueError("'quantize_bias' must be a boolean value.")
        
        # Check signed quantization flag (add default if missing)
        if 'quantize_signed' not in quantization:
            quantization['quantize_signed'] = True  # Default to signed quantization
        else:
            if not isinstance(quantization['quantize_signed'], bool):
                raise ValueError("'quantize_signed' must be a boolean value.")
            
        if 'fold_bn_inference' not in quantization:
            quantization['fold_bn_inference'] = True  # Default to folding BatchNorm during inference
        else:
            if not isinstance(quantization['fold_bn_inference'], bool):
                raise ValueError("'fold_bn_inference' must be a boolean value.")

        if quantization['use_matquant']:
            if 'use_codistillation' not in quantization:
                raise ValueError("Missing 'use_codistillation' key in quantization settings.")
            if not isinstance(quantization['use_codistillation'], bool):
                raise ValueError("'use_codistillation' must be a boolean value.")

            if 'loss_weights' not in quantization:
                raise ValueError("Missing 'loss_weights' key in quantization settings.")
            if not isinstance(quantization['loss_weights'], dict):
                raise ValueError("'loss_weights' must be a dictionary.")

            # Check that loss_weights contains an entry for each target bit
            for bit in quantization['target_bits']:
                if bit not in quantization['loss_weights']:
                    raise ValueError(f"'loss_weights' must contain a weight for each bit in 'target_bits'. Missing weight for {bit}-bit.")
                
                # Check that the weight values are numbers
                if not isinstance(quantization['loss_weights'][bit], (int, float)):
                    raise ValueError(f"Weight value for {bit}-bit must be a number.")
                
                # Weight values should be positive
                if quantization['loss_weights'][bit] < 0:
                    raise ValueError(f"Weight value for {bit}-bit cannot be negative.")
        
        if 'use_qat' not in quantization:
            raise ValueError("Missing 'use_qat' key in quantization settings.")
        if not isinstance(quantization['use_qat'], bool):
            raise ValueError("'use_qat' must be a boolean value.")
        
        if 'fx_mode' not in quantization:
            raise ValueError("Missing 'fx_mode' key in quantization settings.")
        if not isinstance(quantization['fx_mode'], bool):
            raise ValueError("'fx_mode' must be a boolean value.")
        
        if quantization['use_matquant'] or quantization['use_qat']:
            if 'quantized_params_list' not in quantization:
                raise ValueError("Missing 'quantized_params_list' key in quantization settings.")

            quantized_params_list = quantization['quantized_params_list']
            if not isinstance(quantized_params_list, (list, str)):
                raise ValueError("'quantized_params_list' must be a list or a string.")
            
            # Validate that quantized_params_list contains only integers or only strings
            if len(quantized_params_list) > 0:
                all_ints = all(isinstance(x, int) for x in quantized_params_list)
                all_strings = all(isinstance(x, str) for x in quantized_params_list)
                
                if not (all_ints or all_strings):
                    raise ValueError("'quantized_params_list' must contain either all integers or all strings")


    ### Check model settings
    if 'model' in config:
        model = config['model']
        
        if 'name' not in model:
            raise ValueError("Missing 'name' key in model settings.")
        if not isinstance(model['name'], str):
            raise ValueError("'name' must be a string.")
        
        # Expanded list of valid model names
        valid_models = [
            'MLP', 'VGG', 
            'RESNET18', 'RESNET52',
            'TIMM/RESNET18', 'TIMM/RESNET-18', 
            'MICROSOFT/RESNET-18', 'MICROSOFT/RESNET-50',  # Changed from RESNET18 to RESNET-18
            # MobileNet variants
            'GOOGLE/MOBILENET_V1_1.0_224', 'GOOGLE/MOBILENET_V2_1.0_224',
            'TIMM/MOBILENETV1_125.RA4_E3600_R224_IN1K',
            'TIMM/MOBILENETV2_100.RA_IN1K',
            'TIMM/MOBILENETV3_SMALL_100.LAMB_IN1K', 'TIMM/MOBILENETV3_LARGE_100.RA_IN1K',
            'TIMM/MOBILENETV4_CONV_MEDIUM.E500_R256_IN1K',
            # EfficientNet variants
            'GOOGLE/EFFICIENTNET-B0',
            'TIMM/EFFICIENTNET_B0.RA_IN1K'
        ]
        
        if model['name'].upper() not in valid_models:
            # More flexible matching for TIMM/Google models
            name_lower = model['name'].lower()
            is_valid = (
                'mobilenet' in name_lower or 
                'efficientnet' in name_lower or
                'resnet' in name_lower or
                model['name'].upper() in ['MLP', 'VGG', 'RESNET18', 'RESNET52']
            )
            if not is_valid:
                raise ValueError(f"Invalid model name: {model['name']}. Must be one of the supported models or variants.")

        if 'dataset' not in model:
            raise ValueError("Missing 'dataset' key in model settings.")
        if not isinstance(model['dataset'], str):
            raise ValueError("'dataset' must be a string.")
        if model['dataset'].upper() not in ['MNIST', 'FASHION', 'CIFAR10', 'CIFAR100', 'IMAGENETTE', 'IMAGENET', 'IMAGENET1K', 'IMAGENET-1K']:
            raise ValueError("Invalid dataset name. Must be 'MNIST', 'FASHION', 'CIFAR10', 'CIFAR100', 'IMAGENETTE', or 'IMAGENET1K'.")
        
        if 'use_hf' in model and model['use_hf']:
            if not isinstance(model['use_hf'], bool):
                raise ValueError("'use_hf' must be a boolean value.")
            else:
                use_hf = True  # Use Hugging Face model if specified
        else:
            model['use_hf'] = False  # Default to False if not specified
            use_hf = False  # Default to False if not specified

            if 'input_size' not in model:
                raise ValueError("Missing 'input_size' key in model settings.")
            if not isinstance(model['input_size'], int) or model['input_size'] <= 0:
                raise ValueError("'input_size' must be a positive integer.")
            
            if model['name'].upper() == 'MLP':
                if 'hidden_layers' not in model:
                    raise ValueError("Missing 'hidden_layers' key in model settings.")
                if not isinstance(model['hidden_layers'], list):
                    raise ValueError("'hidden_layers' must be a list of dictionaries.")
                
                for layer_config in model['hidden_layers']:
                    if not isinstance(layer_config, dict):
                        raise ValueError("Each hidden layer configuration must be a dictionary.")
                    
                    if 'size' not in layer_config:
                        raise ValueError("Missing 'size' key in hidden layer configuration.")
                    if not isinstance(layer_config['size'], int) or layer_config['size'] <= 0:
                        raise ValueError("'size' must be a positive integer.")
                    
                if 'output_size' not in model:
                    raise ValueError("Missing 'output_size' key in model settings.")
                if not isinstance(model['output_size'], int) or model['output_size'] <= 0:
                    raise ValueError("'output_size' must be a positive integer.")
            
            elif model['name'].upper() == 'VGG':
                if 'num_channels' not in model:
                    raise ValueError("Missing 'num_channels' key in model settings.")
                if not isinstance(model['num_channels'], int) or model['num_channels'] <= 0:
                    raise ValueError("'num_channels' must be a positive integer.")
                
                if 'num_blocks' not in model:
                    raise ValueError("Missing 'num_blocks' key in model settings.")
                if not isinstance(model['num_blocks'], int) or model['num_blocks'] <= 0:
                    raise ValueError("'num_blocks' must be a positive integer.")
                
                if 'kernel_size' not in model:
                    raise ValueError("Missing 'kernel_size' key in model settings.")
                if not isinstance(model['kernel_size'], int) or model['kernel_size'] <= 0:
                    raise ValueError("'kernel_size' must be a positive integer.")
                
                if 'stride' not in model:
                    raise ValueError("Missing 'stride' key in model settings.")
                if not isinstance(model['stride'], int) or model['stride'] <= 0:
                    raise ValueError("'stride' must be a positive integer.")
                
                if 'padding' not in model:
                    raise ValueError("Missing 'padding' key in model settings.")
                if not isinstance(model['padding'], int) or model['padding'] <= 0:
                    raise ValueError("'padding' must be a positive integer.")
                
                if 'linear_layers' not in model:
                    raise ValueError("Missing 'linear_layers' key in model settings.")
                if not isinstance(model['linear_layers'], list):
                    raise ValueError("'linear_layers' must be a list of dictionaries.")
                
                for layer_config in model['linear_layers']:
                    if not isinstance(layer_config, dict):
                        raise ValueError("Each linear layer configuration must be a dictionary.")
                    
                    if 'size' not in layer_config:
                        raise ValueError("Missing 'size' key in linear layer configuration.")
                    if not isinstance(layer_config['size'], int) or layer_config['size'] <= 0:
                        raise ValueError("'size' must be a positive integer.")
                    
                if 'num_classes' not in model:
                    raise ValueError("Missing 'num_classes' key in model settings.")
                if not isinstance(model['num_classes'], int) or model['num_classes'] <= 0:
                    raise ValueError("'num_classes' must be a positive integer.")
                
            elif model['name'].upper() == 'RESNET18' or model['name'].upper() == 'RESNET52':
                if 'num_channels' not in model:
                    raise ValueError("Missing 'num_channels' key in model settings.")
                if not isinstance(model['num_channels'], int) or model['num_channels'] <= 0:
                    raise ValueError("'num_channels' must be a positive integer.")
                
                if 'num_classes' not in model:
                    raise ValueError("Missing 'num_classes' key in model settings.")
                if not isinstance(model['num_classes'], int) or model['num_classes'] <= 0:
                    raise ValueError("'num_classes' must be a positive integer.")
                
                if 'num_blocks' not in model:
                    raise ValueError("Missing 'num_blocks' key in model settings.")
                if not isinstance(model['num_blocks'], list):
                    raise ValueError("'num_blocks' must be a list.")
                for block_count in model['num_blocks']:
                    if not isinstance(block_count, int) or block_count <= 0:
                        raise ValueError("Each value in 'num_blocks' must be a positive integer.")
                    
                if 'base_channels' not in model:
                    raise ValueError("Missing 'base_channels' key in model settings.")
                if not isinstance(model['base_channels'], int) or model['base_channels'] <= 0:
                    raise ValueError("'base_channels' must be a positive integer.")
                
                if 'use_bottleneck' not in model:
                    raise ValueError("Missing 'use_bottleneck' key in model settings.")
                if not isinstance(model['use_bottleneck'], bool):
                    raise ValueError("'use_bottleneck' must be a boolean value.")

    ### Check training settings
    if 'training' in config:
        training = config['training']

        if 'model_dir' not in training:
            raise ValueError("Missing 'model_dir' key in training settings.")
        if not isinstance(training['model_dir'], str):
            raise ValueError("'model_dir' must be a string.")
        
        # Add validation for model_savename
        if 'model_savename' not in training:
            training['model_savename'] = "trained_model"  # Default model name
        elif not isinstance(training['model_savename'], str):
            raise ValueError("'model_savename' must be a string.")
        
        if use_hf:
            # if 'split' not in training:
            #     raise ValueError("Missing 'split' key in training settings.")
            # if not isinstance(training['split'], str):
            #     raise ValueError("'split' must be a string.")
            # if training['split'].upper() not in ['TRAIN', 'TEST', 'VAL']:
            #     raise ValueError("Invalid split name. Must be 'TRAIN', 'TEST', or 'VAL'.")
            
            if 'samples' not in training:
                raise ValueError("Missing 'samples' key in training settings.")
            if not isinstance(training['samples'], int) or training['samples'] <= 0:
                raise ValueError("'samples' must be a positive integer.")
            
        if 'lr_scheduler' not in training:
            raise ValueError("Missing 'lr_scheduler' key in training settings.")
        if not isinstance(training['lr_scheduler'], str):
            raise ValueError("'lr_scheduler' must be a string.")
        if training['lr_scheduler'].upper() not in ['STEP', 'COSINE']:
            raise ValueError("Invalid learning rate scheduler name. Must be 'STEP' or 'COSINE'.")

        if 'optimizer' not in training:
            raise ValueError("Missing 'optimizer' key in training settings.")
        if training['optimizer'].upper() not in ['SGD', 'ADAM', 'ADAMW']:
            raise ValueError("Invalid optimizer name. Must be 'SGD', 'ADAM', or 'ADAMW'.")

        if 'batch_size' not in training:
            raise ValueError("Missing 'batch_size' key in training settings.")        
        if not isinstance(training['batch_size'], int) or training['batch_size'] <= 0:
            raise ValueError("'batch_size' must be a positive integer.")
        
        if 'num_epochs' not in training:
            raise ValueError("Missing 'num_epochs' key in training settings.")
        if not isinstance(training['num_epochs'], int) or training['num_epochs'] < 0:
            raise ValueError("'num_epochs' must be a positive integer.")

        if 'learning_rate' not in training:
            raise ValueError("Missing 'learning_rate' key in training settings.")
        if not isinstance(training['learning_rate'], float) or training['learning_rate'] <= 0:
            raise ValueError("'learning_rate' must be a positive floating point value.")
        
        if 'gamma' not in training:
            raise ValueError("Missing 'gamma' key in training settings.")
        if not isinstance(training['gamma'], float) or training['gamma'] <= 0:
            raise ValueError("'gamma' must be a positive floating point value.")
        
        if 'step_size' not in training:
            raise ValueError("Missing 'step_size' key in training settings.")
        if not isinstance(training['step_size'], int) or training['step_size'] <= 0:
            raise ValueError("'step_size' must be a positive integer.")
        
        if 'momentum' not in training:
            raise ValueError("Missing 'momentum' key in training settings.")
        if not isinstance(training['momentum'], float) or training['momentum'] < 0:
            raise ValueError("'momentum' must be a non-negative floating point value.")
        
        if 'weight_decay' not in training:
            raise ValueError("Missing 'weight_decay' key in training settings.")
        if not isinstance(training['weight_decay'], float) or training['weight_decay'] < 0:
            raise ValueError("'weight_decay' must be a non-negative floating point value.")
        
        # Add validation for dry_iterations
        if 'dry_iterations' not in training:
            training['dry_iterations'] = 0  # Default to 0 if not specified
        elif not isinstance(training['dry_iterations'], int) or training['dry_iterations'] < 0:
            raise ValueError("'dry_iterations' must be a non-negative integer.")
        
        # Add validation for validation split
        if 'val_split' not in training:
            training['val_split'] = 0.0  # Default to 0 (no validation split)
        elif not isinstance(training['val_split'], (int, float)) or training['val_split'] < 0 or training['val_split'] >= 1:
            raise ValueError("'val_split' must be a number between 0 and 1 (exclusive).")
        
        # Add validation for early stopping parameters
        if 'early_stopping' in training:
            early_stopping = training['early_stopping']
            
            if not isinstance(early_stopping, dict):
                raise ValueError("'early_stopping' must be a dictionary.")
            
            # Validate 'enabled' flag
            if 'enabled' not in early_stopping:
                early_stopping['enabled'] = False  # Default to disabled
            elif not isinstance(early_stopping['enabled'], bool):
                raise ValueError("'early_stopping.enabled' must be a boolean value.")
            
            # Only validate other parameters if early stopping is enabled
            if early_stopping['enabled']:
                # Validate 'monitor' parameter - now supports 4 options
                if 'monitor' not in early_stopping:
                    early_stopping['monitor'] = 'val_loss'  # Default to val_loss
                elif not isinstance(early_stopping['monitor'], str):
                    raise ValueError("'early_stopping.monitor' must be a string.")
                elif early_stopping['monitor'] not in ['val_loss', 'val_accuracy', 'test_accuracy', 'epoch_loss']:
                    raise ValueError("'early_stopping.monitor' must be one of: 'val_loss', 'val_accuracy', 'test_accuracy', 'epoch_loss'.")
                
                # Validate 'patience' parameter
                if 'patience' not in early_stopping:
                    early_stopping['patience'] = 5  # Default to 5 epochs
                elif not isinstance(early_stopping['patience'], int) or early_stopping['patience'] <= 0:
                    raise ValueError("'early_stopping.patience' must be a positive integer.")
                
                # Validate 'min_delta' parameter
                if 'min_delta' not in early_stopping:
                    early_stopping['min_delta'] = 0.001  # Default to 0.001
                elif not isinstance(early_stopping['min_delta'], (int, float)) or early_stopping['min_delta'] < 0:
                    raise ValueError("'early_stopping.min_delta' must be a non-negative number.")
                
                # Warn if monitoring validation metrics but no validation split is specified
                if early_stopping['monitor'] in ['val_loss', 'val_accuracy'] and training.get('val_split', 0.0) <= 0:
                    print(f"Warning: Early stopping is monitoring '{early_stopping['monitor']}' but 'val_split' is 0. Early stopping will not work properly without validation data.")
        else:
            # Add default early_stopping config if not present
            training['early_stopping'] = {
                'enabled': False,
                'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.001
            }

    ### Check evaluation settings
    if 'evaluation' in config:
        evaluation = config['evaluation']

        if 'model_path' not in evaluation:
            raise ValueError("Missing 'model_path' key in evaluation settings.")
        if not isinstance(evaluation['model_path'], str):
            raise ValueError("'model_path' must be a string.")
        
        if use_hf:
            # if 'split' not in evaluation:
            #     raise ValueError("Missing 'split' key in evaluation settings.")
            # if not isinstance(evaluation['split'], str):
            #     raise ValueError("'split' must be a string.")
            # if evaluation['split'].upper() not in ['TRAIN', 'TEST', 'VAL']:
            #     raise ValueError("Invalid split name. Must be 'TRAIN', 'TEST', or 'VAL'.")
            
            if 'samples' not in evaluation:
                raise ValueError("Missing 'samples' key in evaluation settings.")
            if not isinstance(evaluation['samples'], int) or evaluation['samples'] <= 0:
                raise ValueError("'samples' must be a positive integer.")
        
        if 'batch_size' not in evaluation:
            raise ValueError("Missing 'batch_size' key in evaluation settings.")
        if not isinstance(evaluation['batch_size'], int) or evaluation['batch_size'] <= 0:
            raise ValueError("'batch_size' must be a positive integer.")
        
        if 'num_iterations' not in evaluation:
            raise ValueError("Missing 'num_iterations' key in evaluation settings.")
        if not isinstance(evaluation['num_iterations'], int) or evaluation['num_iterations'] <= 0:
            raise ValueError("'num_iterations' must be a positive integer.")
        
        if 'mix_and_match_configs' in evaluation:
            mix_and_match_configs = evaluation['mix_and_match_configs']
            
            if not isinstance(mix_and_match_configs, list):
                raise ValueError("'mix_and_match_configs' must be a list.")
            
            for config_entry in mix_and_match_configs:
                if not isinstance(config_entry, dict):
                    raise ValueError("Each entry in 'mix_and_match_configs' must be a dictionary.")
                
                if 'description' not in config_entry:
                    raise ValueError("Missing 'description' key in mix_and_match config entry.")
                if not isinstance(config_entry['description'], str):
                    raise ValueError("'description' must be a string.")
                
                if 'config' not in config_entry:
                    raise ValueError("Missing 'config' key in mix_and_match config entry.")
                if not isinstance(config_entry['config'], dict):
                    raise ValueError("'config' must be a dictionary mapping layer names to bit-widths.")
                
                # Check that all values in config are integers
                for layer, bit_width in config_entry['config'].items():
                    if not isinstance(bit_width, int) or bit_width <= 0:
                        raise ValueError(f"Bit width for layer '{layer}' must be a positive integer.")


