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

## ⚡ Pipelines & CLI Commands

> [!TIP]
> **In-Cluster Execution**: All Python operations can be executed directly inside Kubernetes via:  
> `kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- <command>`  
> To execute locally from your host machine, establish port-forward tunnels (`kubectl port-forward svc/k8s-mlops-postgres 5432:5432` and `kubectl port-forward svc/k8s-mlops-mlflow 5002:5000`).

| Operation | In-Cluster Command (`kubectl exec ... --`) | Local Command | Description |
| :--- | :--- | :--- | :--- |
| **Initial Training** | `python -m src.training.train_xgboost` | `python -m src.training.train_xgboost` | Fits baseline XGBoost model, logs metrics (RMSE, R²), initializes Shapash explainer and registers version 1 in MLflow. |
| **Ray Distributed HPO** | `python -m src.training.hpo_optuna` | `python -m src.training.hpo_optuna` | Distributed Bayesian parameter tuning via Ray workers and PostgreSQL (`optuna`), with automatic fallback. |
| **Closed-Loop Retraining** | `python -m src.training.retrain_pipeline` | `python -m src.training.retrain_pipeline` | Complete autonomous loop: HPO -> XGBoost Retrain -> Champion vs Challenger (gain >= 2%) -> MLflow Promotion -> Hot-Reload. |
| **Webhook Retraining** | N/A (REST API) | `Invoke-RestMethod -Method Post -Uri http://localhost:8000/webhooks/trigger-retrain -Headers @{"x-webhook-secret"="k8s-mlops-super-secure-token"} -ContentType "application/json" -Body '{"source":"manual","event_type":"drift_alert","drift_score":0.42}'` | Asynchronous trigger invoked upon data or concept drift alert. |
| **Drift Monitoring** | `python -m src.monitoring.drift_detector` | `python -m src.monitoring.drift_detector` | Runs Kolmogorov-Smirnov statistical tests and RMSE degradation checks via Evidently AI. |
| **Threshold Calibration** | `python -m src.utils.threshold` | `python -m src.utils.threshold` | Calibrates optimal decision threshold (F1/F2, cost matrix) and plots confusion metrics. |
| **Hot-Reload In-Memory** | `python -m src.utils.api_reloader` | `python -m src.utils.api_reloader` | Triggers immediate hot-reload of active model in memory with zero-downtime health verification. |
| **MLflow Quality Gate** | `python -m src.utils.mlflow_manager` | `python -m src.utils.mlflow_manager` | Evaluates Champion vs Challenger metrics and enforces minimum relative gain (+2%) before alias promotion. |
| **Queue Batch Worker** | `python -m src.utils.process_queue --dir data/queue` | `python -m src.utils.process_queue --dir data/queue` | Consumes queued CSV files and streams predictions to FastAPI `/predict/batch`. |
| **Platform Health Check** | `python -m src.utils.check_status` | `python -m src.utils.check_status` | Audits FastAPI, PostgreSQL databases (`mlops`, `mlflow`, `optuna`), and MLflow tracking readiness. |
| **Full MLOps Reset** | `python -m src.utils.full_reset_mlops` | `python -m src.utils.full_reset_mlops` | Purges MLflow runs/models, resets operational PostgreSQL tables, and bootstraps a fresh Champion baseline. |


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
