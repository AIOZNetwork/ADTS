# DFL Framework: Accelerated Deterministic Topology Scheduling (ADTS)

ADTS extends the Decentralized Federated Learning (DFL) framework with deterministic, role-aware scheduling and early fault-aware topology adaptation to accelerate decentralized FL under heterogeneous and dynamic conditions.

---

# Features

- **Accelerated Deterministic Topology Scheduling (ADTS):** Deterministic, role-aware scheduling with early fault-aware topology adaptation.
- **etcd-based membership coordination:** Inspired by the etcd coordination mechanism used in NVIDIA Dynamo, nodes register their membership through etcd, while the topology server watches membership changes and updates the topology served to nodes.
- Decentralized Federated Learning architecture
- Dynamic topology generation
- Multi-node training orchestration
- Evaluation and inference APIs
- Configurable graph-based communication
- FastAPI-based training services
- Modular node configuration

For fault-aware topology adaptation, deploy etcd clients on the participating nodes and the topology-server host. Configure each node's `membership.etcd_host` and `membership.etcd_port`, and start the topology server with `--etcd-host` and `--etcd-port` pointing to the appropriate etcd endpoint. When membership changes, the topology server can regenerate the topology and notify connected nodes of the new version.

---

### Install Framework Dependencies

```bash
cd aid-dfl-framework
pip install -e .
```

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