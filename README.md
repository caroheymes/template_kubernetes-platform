# k8s_mlops_platform — Cloud-Native MLOps Template for Kubernetes

[![CI Pipeline](https://github.com/caroheymes/template_kubernetes-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/caroheymes/template_kubernetes-platform)
[![Tests](https://img.shields.io/badge/Tests-49%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-v1.28%2B-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB5424)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![dbt](https://img.shields.io/badge/Analytics-dbt-FF694B?logo=dbt&logoColor=white)](https://www.getdbt.com)

A turnkey, production-grade, 100% open-source MLOps template orchestrated natively on **Kubernetes**. Zero vendor lock-in, zero hardcoded project dependencies.

---

## 🌟 Key Capabilities

* **High-Throughput Serving**: Asynchronous **FastAPI** (`/predict`, `/predict/batch`, `/predict/explain`) with Pydantic v2 schemas and Prometheus metrics.
* **Explainable AI (XAI)**: Native **Shapash** integration providing accessible local (waterfall) and global feature importance.
* **Analytics as Code**: **dbt** (`dbt-postgres`) modeling raw prediction logs and features into clean, tested data marts (`mart_inference_kpis`, `mart_prediction_distribution`).
* **Multi-Page Web UI**: **Streamlit** 4-page dashboard:
  1. `App`: Interactive feature input and live inference testing.
  2. `XAI`: Shapash explainability visualizations.
  3. `Analytics`: Performance and operational metrics powered by dbt.
  4. `Webhooks`: Event monitoring and automated retraining triggers.
* **Distributed HPO**: **Optuna** Bayesian optimization backed by PostgreSQL RDB storage and logged directly to **MLflow**.
* **Closed-Loop MLOps**: **Evidently** statistical drift detection (Kolmogorov-Smirnov) triggering automated retraining and Champion vs. Challenger model promotion.
* **Declarative Helm Deployments**: Complete Kubernetes Chart with non-root security compliance (PSS Restricted).

---

## 🏛️ Architecture Overview

```
                                  Ingress NGINX
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  FastAPI Serving               Streamlit Dashboard            MLflow Server
 (Inference & Webhooks)        (Multi-Pages App / XAI)      (Tracking & Registry)
  Port: 8000 /docs             (Analytics & Webhooks)        Port: 5002
           │                     Port: 8503                          │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
             ┌───────────────────────────────────────────────────────┐
             │       PostgreSQL (StatefulSet K8s + Volume PVC)       │
             │   - Base `mlops` : features, predictions, analytics   │
             │   - Base `mlflow` : metadata & Model Registry         │
             │   - Base `optuna` : HPO study trials                  │
             └──────────────────────────┬────────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     S3 / MinIO Object Storage                       Compute & Automation
   - XGBoost models (.json/.ubj)                    - Optuna HPO Worker (Ray/K8s)
   - Shapash SmartExplainer objects                 - Evidently Drift Detector Job
   - Validation & HTML reports                      - dbt Transform Job (K8s CronJob)
```

---

## 🚀 Quickstart (Local Cluster)

### 1. Initialize Cluster
```powershell
# Option A: Docker Desktop (Recommended on Windows — Native WSL2 & GPU support)
# Settings -> Kubernetes -> Check "Enable Kubernetes" -> Apply & restart
kubectl config use-context docker-desktop

# Option B: Minikube (Optimized for Deep Learning: 8 vCPUs, 16 GB RAM & NVIDIA GPU)
minikube start --driver=docker --cpus=8 --memory=16384 --gpus=all
```

### 2. Install KubeRay CRDs
```powershell
kubectl apply --server-side -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v1.1.0"
```

### 3. Build Container Images Locally
```powershell
# Build FastAPI Serving Image
docker build -t mlops-serving:v1.0.0 -f docker/Dockerfile.serving .

# Build Streamlit App & Workers Image
docker build -t mlops-app:v1.0.0 -f docker/Dockerfile.app .
```

### 4. Deploy via Helm
```powershell
helm upgrade --install k8s-mlops ./helm/ `
  --namespace k8s-mlops --create-namespace `
  -f ./helm/values.yaml
```

### 5. Access Services (Port-Forwarding)
```powershell
# FastAPI Swagger Docs
kubectl port-forward svc/k8s-mlops-serving 8000:8000 -n k8s-mlops
# 👉 http://localhost:8000/docs

# Streamlit Multi-Page UI
kubectl port-forward svc/k8s-mlops-streamlit 8503:8503 -n k8s-mlops
# 👉 http://localhost:8503

# MLflow Tracking & Registry
kubectl port-forward svc/k8s-mlops-mlflow 5002:5000 -n k8s-mlops
# 👉 http://localhost:5002
```

---

## ⚡ Kubernetes Operational Workflows & Pipelines

In a cloud-native architecture, all machine learning tasks, transformations, and audits execute inside isolated containers within the Kubernetes cluster:

### 1. In-Cluster ML Jobs (`kubectl run / create job`)
Execute ephemeral tasks inheriting cluster configuration, secrets, and GPU/CPU limits:

```bash
# 1. Baseline Model Training & Initial Registry Bootstrap
kubectl run mlops-train --rm -it --restart=Never --image=ghcr.io/caroheymes/template_kubernetes-platform/app:latest -n k8s-mlops \
  --env-from=configmap/k8s-mlops-config \
  --command -- python -m src.training.train_xgboost

# 2. Distributed Hyperparameter Tuning (Ray + Optuna)
kubectl run mlops-hpo --rm -it --restart=Never --image=ghcr.io/caroheymes/template_kubernetes-platform/app:latest -n k8s-mlops \
  --env-from=configmap/k8s-mlops-config \
  --command -- python -m src.training.hpo_optuna

# 3. Autonomous Closed-Loop Retraining (HPO -> Fit -> Quality Gate -> Hot-Reload)
kubectl run mlops-retrain --rm -it --restart=Never --image=ghcr.io/caroheymes/template_kubernetes-platform/app:latest -n k8s-mlops \
  --env-from=configmap/k8s-mlops-config \
  --command -- python -m src.training.retrain_pipeline

# 4. Data & Concept Drift Analysis (Evidently Report Generation)
kubectl run mlops-drift --rm -it --restart=Never --image=ghcr.io/caroheymes/template_kubernetes-platform/app:latest -n k8s-mlops \
  --env-from=configmap/k8s-mlops-config \
  --command -- python -m src.monitoring.drift_detector

# 5. Immediate Analytics Transformation (from dbt CronJob template)
kubectl create job --from=cronjob/k8s-mlops-dbt-runner dbt-manual-run -n k8s-mlops

# 6. In-Cluster Platform & Multi-Database Health Audit
kubectl run mlops-check --rm -it --restart=Never --image=ghcr.io/caroheymes/template_kubernetes-platform/app:latest -n k8s-mlops \
  --env-from=configmap/k8s-mlops-config \
  --command -- python -m src.utils.check_status
```

### 2. Live Inference & API Consumption (`curl`)
Consume real-time predictions through the FastAPI microservice:

```bash
# Real-Time Unit Inference (< 50ms)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 1.25, "feature_2": -0.45, "feature_3": 2.10, "feature_4": 0.85}}'

# Vectorized Batch Inference
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"features": {"feature_1": 1.25, "feature_2": -0.45, "feature_3": 2.10, "feature_4": 0.85}}]}'

# Explainable Prediction (XAI feature contributions via Shapash)
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 1.25, "feature_2": -0.45, "feature_3": 2.10, "feature_4": 0.85}}'
```

### 3. Operations via Streamlit Portal (`http://localhost:8503`)
For interactive workflows and operational visibility:
* **Page 1 (App)** : Real-time single & batch inference testing, file uploads, and queue streaming.
* **Page 2 (XAI)** : Shapash feature importance and waterfall contribution charts.
* **Page 3 (Analytics)** : dbt performance marts, request throughput, and latency percentiles (P50/P95/P99).
* **Page 4 (Webhooks)** : Drift alert testing, automated retraining triggers, and event log auditing.


---

## 📚 Documentation


* [PRD Specification](docs/prd.md)
* [Technical Architecture Spine](docs/architecture.md)
* [Step-by-Step Launch Guide](docs/launch.md)

---

## 🧪 Testing & Quality Assurance

```powershell
# Run Unit, Integration & API Tests
pytest tests/ -v

# Run Code Formatting and Linting
ruff check .
ruff format --check .

# Validate Helm Templates
helm lint ./helm/
```

---

## 📄 License
MIT License. Free to use, adapt and distribute.
