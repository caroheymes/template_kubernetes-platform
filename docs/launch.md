# Guide de Déploiement & Démarrage — `k8s_mlops_platform`

Ce guide fournit la séquence exacte des commandes PowerShell pour initialiser le cluster Kubernetes à partir de Docker, construire les images conteneurisées, déployer la stack déclarative via Helm et vérifier les services.

---

## 0. Initialisation du Cluster Kubernetes (à partir de Docker)

Choisissez l'une des deux méthodes selon votre configuration :

### Option A : Via Docker Desktop (Recommandé)
1. Ouvrez l'interface graphique **Docker Desktop**.
2. Allez dans **Settings** (icône d'engrenage) > **Kubernetes**.
3. Cochez **Enable Kubernetes** et validez avec **Apply & restart**.
4. Dans PowerShell, vérifiez que le nœud est prêt :
   ```powershell
   kubectl get nodes
   # Doit retourner: docker-desktop ... Ready
   ```

### Option B : Via Minikube (Pilote Docker)
Si vous utilisez Minikube avec le moteur Docker sous Windows :
```powershell
# Suppression propre si un ancien cluster avec un autre driver existe
minikube delete --all

# Démarrage avec le pilote docker et allocation de ressources adaptées
minikube start --driver=docker --cpus=4 --memory=8192

# Vérification
kubectl get nodes
```

---

## 1. Installation des CRDs KubeRay (Une seule fois par cluster)

Pour activer le contrôleur de calcul distribué sans erreur de quota d'annotations :
```powershell
kubectl apply --server-side -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v1.1.0"
```

---

## 2. Construction Locale des Images Docker

Positionnez-vous dans le dossier racine du projet `k8s_mlops_platform` :
```powershell
cd C:\Users\Admin\Desktop\JEDHA\DEMODAY\k8s_mlops_platform

# 1. Construction de l'image de Serving FastAPI & Webhooks
docker build -t mlops-serving:v1.0.0 -f docker/Dockerfile.serving .

# 2. Construction de l'image du Dashboard Streamlit 4-Pages, Ingestion & dbt
docker build -t mlops-app:v1.0.0 -f docker/Dockerfile.app .
```

---

## 3. Validation Statique du Chart Helm

```powershell
# 1. Validation syntaxique
helm lint ./helm/

# 2. Simulation du rendu des manifests Kubernetes
helm template k8s-mlops ./helm/
```

---

## 4. Déploiement Déclaratif via Helm

```powershell
helm upgrade --install k8s-mlops ./helm/ `
  --namespace k8s-mlops --create-namespace `
  -f ./helm/values.yaml
```

---

## 5. Surveillance du Démarrage des Pods

```powershell
# Afficher l'ensemble des ressources créées
kubectl get pods,pvc,svc,statefulset -n k8s-mlops

# Suivre le démarrage en direct (Ctrl + C pour quitter une fois tous les pods en "Running")
kubectl get pods -n k8s-mlops -w
```

---

## 6. Accès aux Interfaces Web (Sans Conflit de Port)

Ouvrez trois fenêtres PowerShell distinctes pour accéder aux services :

### 1. API FastAPI & Swagger Interactive
```powershell
kubectl port-forward svc/k8s-mlops-serving 8000:8000 -n k8s-mlops
```
👉 Ouvrez votre navigateur sur : **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 2. Dashboard Streamlit 4-Pages (App, XAI, Analytics, Webhooks)
```powershell
kubectl port-forward svc/k8s-mlops-streamlit 8503:8503 -n k8s-mlops
```
👉 Ouvrez votre navigateur sur : **[http://localhost:8503](http://localhost:8503)**

### 3. Serveur de Tracking & Model Registry MLflow
```powershell
kubectl port-forward svc/k8s-mlops-mlflow 5002:5000 -n k8s-mlops
```
👉 Ouvrez votre navigateur sur : **[http://localhost:5002](http://localhost:5002)**

---

## 7. Commandes d'Entraînement, Réentraînement & Opérations MLOps

### Option A : Exécution DIRECTEMENT DANS le Cluster Kubernetes (`kubectl exec`)
Puisque le code, les dépendances Python et l'accès direct aux services internes (`postgres:5432`, `mlflow:5000`) sont conteneurisés dans le pod applicatif, utilisez `kubectl exec` :

```powershell
# 1. Entraînement initial baseline dans le cluster (XGBoost + Shapash + MLflow)
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.training.train_xgboost

# 2. Optimisation d'hyperparamètres distribuée (Ray + Optuna)
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.training.hpo_optuna

# 3. Réentraînement complet autonome en boucle fermée (Closed-Loop Retraining)
# (HPO -> Fit XGBoost -> Gain >= 2% -> Promotion @champion -> Hot-Reload)
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.training.retrain_pipeline

# 4. Ingestion batch queue dans le cluster
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.utils.process_queue --dir data/queue

# 5. Diagnostic de santé complet depuis l'intérieur du cluster
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.utils.check_status

# 6. Grand Reset MLOps (Purge & Réinitialisation Contrôlée)
kubectl exec -it deployment/k8s-mlops-streamlit -n k8s-mlops -c streamlit -- python -m src.utils.full_reset_mlops
```

---

### Option B : Déclenchement via Webhook FastAPI (Recommandé en Production)
Déclenche de façon non bloquante la boucle de réentraînement via l'API REST sécurisée :

```powershell
# Depuis la machine hôte (avec port-forward 8000:8000 actif)
$headers = @{ "X-Webhook-Token" = "k8s-mlops-super-secure-token" }
$body = @{
    event = "drift_detected"
    drift_type = "dual_drift_detected"
    features_drifted = @("feature_1", "feature_3")
    psi_score = 0.35
    degradation_pct = 18.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/webhooks/retrain" -Method Post -Headers $headers -Body $body -ContentType "application/json"
```

```bash
# Ou directement en interne depuis un pod du cluster
kubectl exec -it deployment/k8s-mlops-serving -n k8s-mlops -c serving -- curl -X POST "http://localhost:8000/webhooks/retrain" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Token: k8s-mlops-super-secure-token" \
  -d '{"event": "drift_detected", "drift_type": "dual_drift_detected"}'
```

---

### Option C : Exécution Locale (Machine Hôte Windows)
Si vous souhaitez exécuter les scripts directement dans votre terminal local, activez préalablement les tunnels de redirection de ports vers PostgreSQL et MLflow :

```powershell
# Dans 2 terminaux distincts :
kubectl port-forward svc/k8s-mlops-postgres 5432:5432 -n k8s-mlops
kubectl port-forward svc/k8s-mlops-mlflow 5002:5000 -n k8s-mlops

# Puis exécutez dans votre environnement Python local (.venv) :
python -m src.training.train_xgboost
python -m src.training.retrain_pipeline
```

---

## 8. Inspection des Logs & Diagnostic

```powershell
# Logs du microservice FastAPI
kubectl logs deployment/k8s-mlops-serving -n k8s-mlops --tail=100 -f

# Logs du Dashboard Streamlit
kubectl logs deployment/k8s-mlops-streamlit -n k8s-mlops --tail=100 -f

# Logs de la base de données PostgreSQL
kubectl logs statefulset/k8s-mlops-postgres -n k8s-mlops --tail=100 -f

# Logs du serveur MLflow
kubectl logs deployment/k8s-mlops-mlflow -n k8s-mlops --tail=100 -f
```


---

## 9. Désinstallation Propre

```powershell
# Désinstaller la release Helm
helm uninstall k8s-mlops -n k8s-mlops

# Supprimer le namespace et les volumes persistants
kubectl delete namespace k8s-mlops
```

