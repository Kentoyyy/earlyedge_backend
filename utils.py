import torch
import torch.nn as nn
from typing import Type

def load_model(model_path: str, model_class: Type[nn.Module], input_size: int = 512, output_size: int = 2) -> nn.Module:
    """
    Load a PyTorch model from a given path.
    
    Args:
        model_path (str): Path to the saved model file.
        model_class (Type[nn.Module]): The model class to instantiate.
        input_size (int): The size of the input layer. Default is 512.
        output_size (int): The number of output classes. Default is 2.
        
    Returns:
        nn.Module: The loaded PyTorch model.
    """
    # Define the model
    model = model_class(input_size, output_size)
    
    # Load the model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at path: {model_path}")
        raise
    except RuntimeError as e:
        print(f"Error loading model state: {e}")
        raise
    
    return model

# Example Model Definition
class SimpleDysgraphiaModel(nn.Module):
    def __init__(self, input_size: int = 512, output_size: int = 2):
        super(SimpleDysgraphiaModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)

# Usage Example
# model = load_model("path_to_model.pth", SimpleDysgraphiaModel)
