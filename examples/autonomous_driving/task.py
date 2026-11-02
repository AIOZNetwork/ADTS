"""Define FADNet training and testing tasks.
Includes functions for data loading, model training, evaluation, and weight management.
"""

import torch
import torch.optim as optim
import tqdm
from collections import OrderedDict

from examples.autonomous_driving.dataloaders.complex_driving import get_iterator_complex_driving
from examples.autonomous_driving.dataloaders.driving import get_iterator_driving

def get_iterator(name, path, device, batch_size):
    """Get data iterator based on dataset name.
    
    Args:
        name: Name of the dataset.
        path: Path to the dataset.
        device: Device to load data onto.
        batch_size: Batch size for the iterator.
    
    Returns:
        Data iterator for the specified dataset.
    """
    if name == "driving_udacity":
        return get_iterator_driving(path, device, batch_size=batch_size)
    elif "driving_gazebo" in name or "driving_carla" in name:
        return get_iterator_complex_driving(path, device, batch_size=batch_size)
    else:
        raise NotImplementedError
    
def get_optimizer(optimizer_name, net, lr_initial=1e-3):
    """Get optimizer based on the specified name.

    Args:
        optimizer_name: Name of the optimizer ("adam" or "sgd").
        net: The neural network model.
        lr_initial: Initial learning rate.
    
    Returns:
        Configured optimizer.
    """
    if optimizer_name == "adam":
        return optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr_initial, weight_decay=1e-5)

    elif optimizer_name == "sgd":
        return optim.SGD([param for param in net.parameters() if param.requires_grad], lr=lr_initial)
    else:
        raise NotImplementedError("Other optimizer are not implemented")

def fit_batch(net, iterator, optimizer, criterion, device):
    """Train the model on a single batch from the iterator.
    
    Args:
        net: The neural network model.
        iterator: Data iterator.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to perform training on.
    
    Returns:
        Loss value for the batch.
    """
    # Switch to train mode
    net.train()

    # Get a single batch. At each fit batch, initialize a new iterator randomly and get the first batch
    x, y = next(iter(iterator))

    # Move data to device
    x = x.to(device)
    y = y.unsqueeze(-1).to(device)
    y = y.to(x.dtype)

    # Perform forward and backward pass
    optimizer.zero_grad()

    predictions = net(x)

    loss = criterion(predictions, y)

    loss.backward()

    optimizer.step()

    batch_loss = loss.item()

    # Return the batch loss value
    return batch_loss

def fit_iterator_one_epoch(net, iterator, optimizer, criterion, device):
    """Train the model for one epoch over the entire iterator.
    
    Args:
        net: The neural network model.
        iterator: Data iterator.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to perform training on.
    
    Returns:
        Average loss over the epoch.
    """
    epoch_loss = 0

    # Switch to train mode
    net.train()

    # Iterate over all batches in the iterator
    for x, y in iterator:
        # Perform forward and backward pass
        optimizer.zero_grad()
        x = x.to(device)
        y = y.unsqueeze(-1).to(device)
        y = y.to(x.dtype)
        predictions = net(x)

        loss = criterion(predictions, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    # Return average loss over the epoch
    return epoch_loss / len(iterator)

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
    net.to(device)  # move model to GPU if available

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer("adam", net, lr_initial=1e-3)
    # switch to train mode
    net.train()
    # Perform training based on fit_type
    running_loss = 0.0
    if fit_type == "epoch":
        for _ in range(epochs):
            epoch_loss = fit_iterator_one_epoch(net, trainloader, optimizer, criterion, device)
            running_loss += epoch_loss
        num_samples = len(trainloader.dataset) * epochs
    elif fit_type == "batch":
        for _ in range(epochs):
            batch_loss = fit_batch(net, trainloader, optimizer, criterion, device)
            running_loss += batch_loss
        num_samples = trainloader.batch_size * epochs
    else:
        raise NotImplementedError("Other fit type are not implemented")
    # Calculate average training loss
    avg_trainloss = running_loss / epochs
    # Return average training loss and number of samples used
    return avg_trainloss, num_samples

def RMSE(y_preds, y_trues):
    """Calculate Root Mean Square Error (RMSE) between predictions and true values.
    
    Args:
        y_preds: Predicted values.
        y_trues: True values.
    
    Returns:
        RMSE value.
    """
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_preds - y_trues)**2))
    
def test(net, testloader, device):
    """Evaluate the model on the test set.
    
    Args:
        net: The neural network model.
        testloader: Data loader for test data.
        device: Device to perform evaluation on.
    
    Returns:
        Average test loss and RMSE over the test set.
    """
    # Define loss function and metric
    criterion = torch.nn.MSELoss()
    metric = RMSE
    # Perform evaluation
    epoch_loss = 0
    epoch_acc = 0

    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm.tqdm(testloader)):
            x = x.to(device)
            y = y.unsqueeze(-1).to(device)
            y = y.to(x.dtype)
            predictions = net(x)

            loss = criterion(predictions, y)

            acc = metric(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # Return average test loss and RMSE
    return epoch_loss / len(testloader), epoch_acc / len(testloader)

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