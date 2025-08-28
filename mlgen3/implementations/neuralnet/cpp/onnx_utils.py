import os
import shutil


def deploy_onnx_model(model, output_path, materializer_class, **kwargs):
    """
    Helper function to deploy an ONNX model using the appropriate materializer.

    Args:
        model: The MLGen3 model or implementation
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
    print(f"Model materialized at: {output_path}")

    # Copy the ONNX model to the deployment directory if available
    if hasattr(model, 'onnx_path') and os.path.exists(model.onnx_path):
        shutil.copy2(model.onnx_path, os.path.join(output_path, os.path.basename(model.onnx_path)))

    return materializer
