import torch

class StdScalerTransform(torch.nn.Module):
    def __init__(self, mean, std):
        """
        Custom transform to normalize signals using z-score normalization.

        Args:
            mean (list or torch.Tensor): Mean of each feature (size: num_features).
            std (list or torch.Tensor): Standard deviation of each feature (size: num_features).
    
        Raises:
            ValueError: If `mean` and `std` are not the same length.
        """
        super().__init__()

        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length.")
        
        self.mean = torch.tensor(mean).to(torch.float)
        self.std = torch.tensor(std).to(torch.float)

    def forward(self, tensor: torch.Tensor):
        """
        Apply normalization to the sample.

        Args:
            tensor (torch.Tensor): Input sample of shape (signal_length, num_features) 
                                   or (batch_size, signal_length, num_features).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as the input.
        """
        # Check the last dimension matches the number of features
        if tensor.shape[-1] != len(self.mean):
            raise ValueError(
                f"The number of features in the input ({tensor.shape[-1]}) does not match "
                f"the length of mean/std ({len(self.mean)})."
            )
        
        return (tensor - self.mean) / (self.std + 1e-8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class MinMaxScalerTransform(torch.nn.Module):
    def __init__(self, min, max):
        """
        Transform to apply Min-Max scaling to input data.

        Args:
            min (list or torch.Tensor): Minimum values for each feature (size: num_features).
            max (list or torch.Tensor): Maximum values for each feature (size: num_features).
        
        Raises:
            ValueError: If `min` and `max` are not the same length.
        """
        super().__init__()

        if len(min) != len(max):
            raise ValueError("min and max must have the same length.")
        
        self.min = torch.tensor(min).to(torch.float)
        self.max = torch.tensor(max).to(torch.float)
        self.range = self.max - self.min

    def forward(self, tensor: torch.Tensor):
        """
        Apply min-max scaling to the input sample.

        Args:
            tensor (torch.Tensor): Input tensor of shape (signal_length, num_features) 
                                   or (batch_size, signal_length, num_features).
        
        Returns:
            torch.Tensor: Scaled tensor of the same shape as the input.
        """
        # Check the last dimension matches the number of features
        if tensor.shape[-1] != len(self.min):
            raise ValueError(
                f"The number of features in the input ({tensor.shape[-1]}) does not match "
                f"the length of min/max ({len(self.min)})."
            )
        
        return (tensor - self.min) / (self.range + 1e-8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"
    
__all__ = [
    "MinMaxScalerTransform",
    "StdScalerTransform",
]