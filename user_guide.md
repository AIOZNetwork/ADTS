# DFL Framework User Guide

## Project Structure

A project using the DFL Framework should follow the structure below:

```text
project/
├── configs/
│   ├── node-0.yaml
│   ├── node-1.yaml
│   ├── ...
│   ├── graph.yaml
│   ├── cluster.yaml
│   └── graph_generation_config.json
├── client.py
└── ...
```

## Configuration Files

### `configs/node-i.yaml`

Each node in the DFL framework is configured through a `node-i.yaml` file.

Example:

```yaml
node_id: node-0
# Whether to obtain the communication topology from a topology server
has_topology: false
# Address of the topology server (used only when has_topology=true)
topology_url: 10.0.0.224:7000
flower:
  # Optional artificial communication delay (seconds)
  delay:
  # Optional node name
  name:
  # Local training batch size
  batch_size: 128
  # Number of local epochs per communication round
  local_epochs: 1
  # Root path for saving checkpoints
  ckpt_root: checkpoints
  # Save a checkpoint every N rounds
  ckpt_round: 100
  # Training dataset
  train_path: benchmark/MultigraphFL/data/femnist/gaia/train/0.pkl
  # Validation/Test dataset
  test_path: benchmark/MultigraphFL/data/femnist/gaia/test/test.pkl

peer:
  # Address for this node's peer service
  host: 0.0.0.0
  # Listening port
  port: 9010
  # Total number of federated training rounds
  rounds: 100

client:
  # Python module containing the client implementation
  module: examples.MultigraphFL.client
  # Client class name (must inherit from DFLClient)
  class: MyClient
```

### Notes

> **Important**
> The client constructor **must** assign the checkpoint root:
> ```python
> self.ckpt_root = config["flower"]["ckpt_root"]
> ```
> The framework uses this value to initialize the checkpoint directory for the current node. When saving checkpoints inside the client, **always use** `self.checkpoint_dir` instead of `self.ckpt_root`.

Example:
```python
if current_round % config["flower"]["ckpt_round"] == 0:
    self.model.save(self.checkpoint_dir / f"{self.node_id}.pth")
```

---

### `configs/graph.yaml`

Defines the communication graph between nodes.

Example:

```yaml
'1':
  node-0:
    recv_ids:
    - node-1
    send_ids:
    - node-1
  node-1:
    recv_ids:
    - node-0
    - node-2
    send_ids:
    - node-0
    - node-2
  node-2:
    recv_ids:
    - node-1
    send_ids:
    - node-1
```

---

### `configs/cluster.yaml`

Defines the addresses, weights of all nodes in the cluster.

Example:

```yaml
node-0:
  weight: 0.5
  address: 10.0.0.224:9010
node-1:
  weight: 0.5
  address: 10.0.0.224:9011
node-2:
  weight: 0.5
  address: 10.0.0.224:9012
```

---

### `configs/graph_generation_config.json`

This optional file is used to automatically generate the communication graph instead of using the default topology.

Example:

```json
{
  "model_size": 100000000.0,
  "num_rounds": 100,
  "path_to_topology": "custom_data/gaia.gml"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_size` | `float` | Model size in **bits**. This value is calculated as the number of model parameters multiplied by the size of the parameter data type. |
| `num_rounds` | `int` | Number of federated training rounds used for topology generation/simulation. |
| `path_to_topology` | `str` | Path to the topology file containing the communication graph definition. The file can be a graph format such as `.gml`. |


If this file exists, the framework generates the communication graph based on the provided configuration instead of using the default graph.

---

# Implementing `client.py`

Users must implement a client class that inherits from `DFLClient`.

```python
from app.client.base import DFLClient

class Client(DFLClient):
    ...
```

The framework will invoke the methods below during training and inference.

---

## `__init__`

```python
def __init__(self, config: dict):
```

### Purpose

Initialize all resources required by the node, including:

- Model
- Optimizer
- Training DataLoader
- Testing DataLoader
- Any additional components

Example:

```python
class Client(DFLClient):

    def __init__(self, config):

        self.config = config

        self.model = ...
        self.optimizer = ...

        self.train_loader = ...
        self.test_loader = ...

        self.ckpt_root = config["flower"]["ckpt_root"]
```

> **Important**
>
> The following line **must** be included:
>
> ```python
> self.ckpt_root = config["flower"]["ckpt_root"]
> ```
>
> The framework uses this directory to save model checkpoints.

---

## `get_parameters`

```python
def get_parameters(self, config: Dict[str, Any]) -> list[np.ndarray]:
```

Returns the model parameters as a list of NumPy arrays.

These parameters will be exchanged with neighboring nodes during distributed federated learning.

---

## `set_parameters`

```python
def set_parameters(self, parameters: list[np.ndarray]) -> None:
```

Loads model parameters received from the framework into the local model.

---

## `fit`

```python
def fit(self, parameters: list[np.ndarray], config: Dict[str, Any]) -> Tuple[list[np.ndarray], int, Dict[str, Any]]
```

Performs local training.

A typical implementation consists of:

1. Load received parameters.
2. Train the model locally.
3. Retrieve updated parameters.
4. Return training results.

Return value:

```python
( parameters, num_examples, metrics )
```

Where:

- `parameters`: Updated model parameters.
- `num_examples`: Number of training samples.
- `metrics`: Dictionary containing training metrics.

Example:

```python
return (self.get_parameters({}), len(self.train_loader.dataset),
    {
        "loss": train_loss,
        "accuracy": train_accuracy,
    },
)
```

---

## `evaluate`

```python
def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]
```

Evaluates the model on the local test dataset.

Return value:

```python
(loss, num_examples, metrics)
```

Example:

```python
return (loss, len(self.test_loader.dataset),
    { "accuracy": accuracy, },
)
```

---

## `inference`

```python
def inference(self, parameters: list[np.ndarray], inputs: bytes) -> np.ndarray:
```

Runs inference for API requests.

Arguments:

- `parameters`: Model parameters.
- `inputs`: Serialized input data.

Returns:

```python
np.ndarray
```

containing the prediction results.

---

## `run_inference`

```python
def run_inference(self, weight_path: Path, input_path: Path, output_path: Path) -> None:
```

Runs inference locally without using the API.

Arguments:

| Parameter | Description |
|-----------|-------------|
| `weight_path` | Path to the model checkpoint |
| `input_path` | Input data path |
| `output_path` | Directory or file for prediction results |

Typical workflow:

1. Load checkpoint from `weight_path`.
2. Read input data from `input_path`.
3. Run inference.
4. Save predictions to `output_path`.

---

# Implementation Checklist

Before running the DFL Framework, ensure that:

- [ ] A `configs/` directory exists.
- [ ] `node-i.yaml` is created for every node.
- [ ] `cluster.yaml` is provided.
- [ ] `graph.yaml` is provided, or `graph_generation_config.json` is available for automatic graph generation.
- [ ] `client.py` implements a class that inherits from `DFLClient`.
    - [ ] `__init__` initializes the model, optimizer, and DataLoaders.
        - [ ] `self.ckpt_root = config["flower"]["ckpt_root"]` is set correctly.
    - [ ] `get_parameters()` is implemented.
    - [ ] `set_parameters()` is implemented.
    - [ ] `fit()` is implemented.
    - [ ] `evaluate()` is implemented.
    - [ ] `inference()` is implemented.
    - [ ] `run_inference()` is implemented.