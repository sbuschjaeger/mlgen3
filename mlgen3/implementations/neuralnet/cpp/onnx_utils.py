def deploy_onnx_model(model, output_path, materializer_class, **kwargs):
    """
    Helper function to deploy an ONNX model using the appropriate materializer.
    
    Args:
        model: The MLGen3 model
        output_path: Path where the generated files will be saved
        materializer_class: The materializer class to use (e.g., LinuxStandalone)
        **kwargs: Additional arguments for the materializer
    
    Returns:
        The configured materializer instance
    """
    # Make sure we're using the ONNX Makefile
    kwargs['use_onnx'] = True
    
    # Create the materializer
    materializer = materializer_class(
        model,
        **kwargs
    )
    
    # Generate the files
    materializer.materialize(output_path)
    
    return materializer
