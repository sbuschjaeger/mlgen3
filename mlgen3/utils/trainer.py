"""Model training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import os


class ModelTrainer:
    """Trainer class for training PyTorch models with MatQuant."""
    
    def __init__(
        self,
        model: nn.Module,
        mq_model,  # MatQuant wrapper
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            mq_model: MatQuant wrapper around the model
            config: Configuration dictionary
            device: Device to train on (default: auto-detect)
        """
        self.model = model
        self.mq_model = mq_model
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
    
    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        training_config = self.config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'sgd').lower()
        lr = training_config.get('learning_rate', 0.01)
        momentum = training_config.get('momentum', 0.9)
        weight_decay = training_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config."""
        training_config = self.config.get('training', {})
        scheduler_name = training_config.get('lr_scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            step_size = training_config.get('step_size', 5)
            gamma = training_config.get('gamma', 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            epochs = training_config.get('num_epochs', 10)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None
    
    def _setup_criterion(self):
        """Setup loss criterion."""
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        batch_size: int,
        epoch: int,
        total_epochs: int
    ) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = (len(train_x) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(train_x), batch_size), 
                     desc=f"Epoch {epoch+1}/{total_epochs}"):
            inputs = train_x[i:i+batch_size].to(self.device)
            targets = train_y[i:i+batch_size].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Multi-precision forward pass
            outputs = self.mq_model.multi_precision_forward(inputs)
            
            # Calculate weighted loss
            loss, individual_losses = self.mq_model.matquant_loss(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        # Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return running_loss / num_batches
    
    def train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: Optional[torch.Tensor] = None,
        test_y: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None
    ) -> Tuple[nn.Module, Any]:
        """
        Train the model.
        
        Args:
            train_x: Training inputs
            train_y: Training labels
            test_x: Test inputs (optional, for evaluation during training)
            test_y: Test labels (optional)
            num_epochs: Number of epochs (overrides config if provided)
        
        Returns:
            Tuple of (trained_model, mq_model)
        """
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', 64)
        epochs = num_epochs if num_epochs else training_config.get('num_epochs', 1)
        
        print(f"\nTraining model for {epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_x, train_y, batch_size, epoch, epochs)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluate if test data provided
            if test_x is not None and test_y is not None:
                self._evaluate_during_training(test_x, test_y, batch_size)
        
        # Save model
        self._save_model()
        
        # Move back to CPU for compatibility
        self.model = self.model.cpu()
        if hasattr(self.mq_model, 'model'):
            self.mq_model.model = self.mq_model.model.cpu()
        
        return self.model, self.mq_model
    
    def _evaluate_during_training(
        self,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        batch_size: int
    ):
        """Evaluate model during training."""
        self.model.eval()
        self.mq_model.eval()
        
        quant_config = self.config.get('quantization', {})
        target_bits = quant_config.get('target_bits', [8, 4, 2])
        
        with torch.no_grad():
            accuracies = {}
            for bits in target_bits:
                correct = 0
                total = 0
                
                for i in range(0, len(test_x), batch_size):
                    inputs = test_x[i:i+batch_size].to(self.device)
                    targets = test_y[i:i+batch_size].to(self.device)
                    
                    outputs = self.mq_model.forward_with_quant(inputs, bits)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
                accuracies[bits] = 100 * correct / total
        
        for bits, acc in accuracies.items():
            print(f"  {bits}-bit Accuracy: {acc:.2f}%")
    
    def _save_model(self):
        """Save the trained model."""
        training_config = self.config.get('training', {})
        model_dir = training_config.get('model_dir', './models')
        model_savename = training_config.get('model_savename', 'model')
        
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_savename}.pt")
        
        # Save on CPU
        model_cpu = self.model.cpu()
        torch.save(model_cpu.state_dict(), model_path)
        print(f"\nSaved model to {model_path}")
        
        # Move back to device
        self.model = self.model.to(self.device)
