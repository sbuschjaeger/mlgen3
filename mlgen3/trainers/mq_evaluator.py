"""Model evaluation utilities."""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional


class ModelEvaluator:
    """Evaluator class for testing PyTorch models with MatQuant."""
    
    def __init__(
        self,
        mq_model,  # MatQuant wrapper
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            mq_model: MatQuant wrapper around the model
            config: Configuration dictionary
            device: Device to evaluate on (default: auto-detect)
        """
        self.mq_model = mq_model
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def extract_and_test_models(
        self,
        test_x: torch.Tensor,
        test_y: torch.Tensor
    ) -> Dict[str, nn.Module]:
        """
        Extract models at different bit-widths and test them.
        
        Args:
            test_x: Test inputs
            test_y: Test labels
        
        Returns:
            Dictionary mapping bit-widths to extracted models
        """
        eval_config = self.config.get('evaluation', {})
        batch_size = eval_config.get('batch_size', 128)
        quant_config = self.config.get('quantization', {})
        target_bits = quant_config.get('target_bits', [8, 4, 2])
        
        # Extract models with different bit-widths
        print("\nExtracting models with different bit-widths...")
        extracted_models = {}
        for bits in target_bits:
            extracted_models[bits] = self.mq_model.extract_model(bits).to(self.device)
        
        # Test extracted models
        print("\nTesting extracted models...")
        for bits, ext_model in extracted_models.items():
            accuracy = self._test_model(ext_model, test_x, test_y, batch_size)
            print(f"Extracted {bits}-bit model accuracy: {accuracy:.2f}%")
        
        print("")
        
        # Move models back to CPU
        for bits in extracted_models:
            extracted_models[bits] = extracted_models[bits].cpu()
        
        return extracted_models
    
    def test_mix_and_match(
        self,
        mix_config: Dict[str, int],
        test_x: torch.Tensor,
        test_y: torch.Tensor
    ) -> nn.Module:
        """
        Create and test a mix-and-match model.
        
        Args:
            mix_config: Dictionary mapping layer names to bit-widths
            test_x: Test inputs
            test_y: Test labels
        
        Returns:
            Mix-and-match model
        """
        eval_config = self.config.get('evaluation', {})
        batch_size = eval_config.get('batch_size', 128)
        
        # Create mix-and-match model
        mix_model = self.mq_model.mix_and_match(mix_config).to(self.device)
        
        # Test mix-and-match model
        accuracy = self._test_model(mix_model, test_x, test_y, batch_size)
        print(f"Mix-and-match model accuracy: {accuracy:.2f}%")
        
        print("")
        
        # Move back to CPU
        mix_model = mix_model.cpu()
        
        return mix_model
    
    def _test_model(
        self,
        model: nn.Module,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        batch_size: int
    ) -> float:
        """
        Test a single model.
        
        Returns:
            Accuracy as percentage
        """
        model.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            
            for i in range(0, len(test_x), batch_size):
                inputs = test_x[i:i+batch_size].to(self.device)
                targets = test_y[i:i+batch_size].to(self.device)
                
                outputs = model(inputs)
                
                # Print logits for the first element
                if i == 0:
                    print(f"\nFirst element logits: {outputs[0].cpu().numpy()}")
                
                _, predicted = torch.max(outputs, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if i==0:
                    print(f"First element predicted: {predicted[0].item()}, target: {targets[0].item()}")
            
            accuracy = 100 * correct / total
        
        return accuracy
