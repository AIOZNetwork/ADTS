"""Define FADNet training and testing tasks.
Includes functions for data loading, model training, evaluation, and weight management.
"""

import torch
from collections import OrderedDict

def train(net, trainloader, epochs, device):
    """Train the model on the training set.
    
    Args:
        net: The neural network model.
        trainloader: Data loader for training data.
        epochs: Number of epochs to train.
        device: Device to perform training on.
    
    Returns:
        Average training loss and number of samples used.
    """
    fit_type = "batch"  # or "epoch"
    # Perform training based on fit_type
    running_loss = 0.0
    if fit_type == "epoch":
        for _ in range(epochs):
            epoch_loss, _ = net.fit_iterator_one_epoch(trainloader)
            running_loss += epoch_loss
        num_samples = len(trainloader.dataset) * epochs
    elif fit_type == "batch":
        for _ in range(epochs):
            batch_loss, _ = net.fit_batch(trainloader)
            running_loss += batch_loss
        num_samples = trainloader.batch_size * epochs
    else:
        raise NotImplementedError("Other fit type are not implemented")
    # Calculate average training loss
    avg_trainloss = running_loss / epochs
    # Return average training loss and number of samples used
    return avg_trainloss, num_samples
    
def test(net, testloader, device):
    """Evaluate the model on the test set.
    
    Args:
        net: The neural network model.
        testloader: Data loader for test data.
        device: Device to perform evaluation on.
    
    Returns:
        Average test loss and RMSE over the test set.
    """
    # Perform evaluation
    loss, metric, _, _ = net.evaluate_iterator(testloader)
    return loss, metric

def get_weights(net):
    """Get model weights as a list of NumPy arrays.
    
    Args:
        net: The neural network model.
        
    Returns:
        List of model weights as NumPy arrays.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays to the model.
    
    Args:
        net: The neural network model.
        parameters: List of model weights as NumPy arrays.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)