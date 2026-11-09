# DFL Framework: Accelerated Deterministic Topology Scheduling (ADTS)

ADTS extends the Decentralized Federated Learning (DFL) framework with deterministic, role-aware scheduling and early fault-aware topology adaptation to accelerate decentralized FL under heterogeneous and dynamic conditions.

## Contents

- [Using DFL Framework in Your Project](#using-dfl-framework-in-your-project)
- [Features](#features)
- [Install Framework Dependencies](#install-framework-dependencies)
- [ETCD Server](#etcd-server)
- [Topology Server](#topology-server)
- [Start DFL Nodes (Silos)](#start-dfl-nodes-silos)
- [Training API Service](#training-api-service-use-when-run-with-topology-server)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [License](#license)

---

## Using DFL Framework in Your Project

Before running the framework for your own project, follow the
[User Guide](./user_guide.md) to configure your nodes and communication topology,
and implement a client tailored to your model and dataset.

---

## Features

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

## Install Framework Dependencies

```bash
cd ADTS/
pip install -e .
```

## ETCD Server
```bash
cd ADTS/etcd_server
docker compose up -d
```

## Topology Server

The topology server is responsible for generating or loading communication graphs for training.

- Generate Topology Dynamically

```bash
cd ADTS/

dfl-topology-server \
  --mode generate \
  --graph-generation-config ./examples/MultigraphFL/custom_config/amazon_us/graph_generation_config.json \
  --cluster-config ./examples/MultigraphFL/custom_config/amazon_us/cluster.yaml
```

- Load Existing Graph Topology

```bash
cd ADTS/

dfl-topology-server \
  --mode default \
  --graph-config ./examples/MultigraphFL/custom_config/amazon_us/graph.yaml \
  --cluster-config ./examples/MultigraphFL/custom_config/amazon_us/cluster.yaml
```

---

## Start DFL Nodes (Silos)

Each silo represents an independent federated learning participant.

- Example

```bash
cd ADTS/

PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/amazon_us/node-0.yaml
PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/amazon_us/node-1.yaml
...
PYTHONPATH=. dfl-node --local-node-config ./examples/MultigraphFL/custom_config/amazon_us/node-21.yaml
# Run each node in a separate terminal session or container.
```

---

## Training API Service (use when run with topology server)

Start the FastAPI training service:

```bash
cd ADTS/src/app/user

uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Training

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

## Evaluation

Evaluate a trained model on selected nodes.

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

## Inference

Run inference using a trained model.

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

## License

This project is licensed under the MIT License.

Copyright (c) 2026 AIOZ Network. See [LICENSE](./LICENSE) for the full license text.