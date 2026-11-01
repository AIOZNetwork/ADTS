# DFL Framework

Decentralized Federated Learning (DFL) framework for decentralized model training, topology orchestration, evaluation, and inference across silos.

---

# Features

- Decentralized Federated Learning architecture
- Dynamic topology generation
- Multi-node training orchestration
- Evaluation and inference APIs
- Configurable graph-based communication
- FastAPI-based training services
- Modular node configuration

---

### Install Framework Dependencies

```bash
cd aid-dfl-framework
pip install -e .
```

---
# Run Without Topology Server

The framework can also run **without starting the topology server**. In this mode, each node loads a predefined communication graph from the configuration and begins training immediately. During training, nodes automatically exchange model updates with their neighbors according to the graph configuration.

### 1. Disable Topology Service

In each node configuration file (`node-i.yaml`), set:

```yaml
has_topology: false
```

### 2. Start a Node

Run each node using its corresponding configuration:

- Generate Topology Dynamically
```bash
PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/gaia/node-0.yaml --graph-mode generate   --graph-generation-config ./examples/MultigraphFL/custom_config/gaia/graph_generation_config.json --cluster-config ./examples/MultigraphFL/custom_config/gaia/cluster.yaml
```

- Load Existing Graph Topology
```bash
PYTHONPATH=. dfl-node \
  --local-node-config ./examples/MultigraphFL/custom_config/gaia/node-0.yaml \
  --graph-config ./examples/MultigraphFL/custom_config/gaia/graph.yaml \
  --cluster-config ./examples/MultigraphFL/custom_config/gaia/cluster.yaml
```

Repeat the command for all nodes by replacing `node-0.yaml` with the corresponding node configuration (e.g., `node-1.yaml`, `node-2.yaml`, etc.).

### Local Inference

After training, you can perform inference locally on a node using a trained checkpoint. Only the checkpoint directory needs to be specified via `--weight-path`.

```bash
PYTHONPATH=. dfl-node \
  --mode inference \
  --weight-path checkpoints/100/ \
  --input-dir debug/inputs/ \
  --output-dir debug/outputs/
```

| Argument | Description |
|----------|-------------|
| `--weight-path` | Path to the trained model checkpoint directory. |
| `--input-dir` | Directory containing input data for inference. |
| `--output-dir` | Directory where inference results will be saved. |

---

# Topology Server

The topology server is responsible for generating or loading communication graphs for training.

### Generate Topology Dynamically

```bash
cd aid-dfl-framework

dfl-topology-server \
  --mode generate \
  --graph-generation-config ./examples/MultigraphFL/custom_config/gaia/graph_generation_config.json \
  --cluster-config ./examples/MultigraphFL/custom_config/gaia/cluster.yaml
```

### Load Existing Graph Topology

```bash
cd aid-dfl-framework

dfl-topology-server \
  --mode default \
  --graph-config ./examples/MultigraphFL/custom_config/gaia/graph.yaml \
  --cluster-config ./examples/MultigraphFL/custom_config/gaia/cluster.yaml
```

---

# Start DFL Nodes (Silos)

Each silo represents an independent federated learning participant.

### Example

```bash
cd aid-dfl-framework

PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/gaia/node-0.yaml
PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/gaia/node-1.yaml
PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/gaia/node-2.yaml
```

> Run each node in a separate terminal session or container.

---

# Training API Service (use when run with topology server)

Start the FastAPI training service:

```bash
cd aid-dfl-framework/src/app/user

uvicorn api:app --host 0.0.0.0 --port 8000
```

### API Documentation

After startup:

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

---

# Training

Trigger training:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'accept: */*' \
  -H 'Authorization: Bearer eyJ...' \
  -H 'Content-Type: application/json' \
  -d '{
  "round_id": 0
}'
```

---

# Evaluation

Evaluate a trained model on selected nodes.

### Request

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H 'accept: */*' \
  -H 'Authorization: Bearer eyJ...' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_key": "100",
    "nodes": ["node-1", "node-10"]
  }'
```

---

# Inference

Run inference using a trained model.

### Request

```bash
curl -X POST "http://localhost:8000/inference" \
  -H 'accept: */*' \
  -H 'Authorization: Bearer eyJ...' \
  -H 'Content-Type: application/json' \
  -d '{
    "data_b64": "...=",
    "model_key": "100",
    "nodes": ["node-1", "node-10"]
  }'
```

---

# License

Internal / Proprietary