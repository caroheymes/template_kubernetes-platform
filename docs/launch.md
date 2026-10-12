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

# Démarrage avec le pilote docker, 8 cœurs, 16 Go de RAM et support GPU NVIDIA
minikube start --driver=docker --cpus=8 --memory=16384 --gpus=all

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

### Option A : Exécution Isolé en Cluster (Pods Éphémères & Jobs K8s)
Pour exécuter les opérations lourdes sans impacter la mémoire ou la disponibilité des pods web applicatifs :

```powershell
# 1. Entraînement initial baseline dans le cluster (XGBoost + Shapash + MLflow)
kubectl run mlops-train --rm -it --restart=Never --image=mlops-app:v1.0.0 -n k8s-mlops `
  --env-from=configmap/k8s-mlops-config `
  --command -- python -m src.training.train_xgboost

# 2. Optimisation d'hyperparamètres distribuée (Ray + Optuna)
kubectl run mlops-hpo --rm -it --restart=Never --image=mlops-app:v1.0.0 -n k8s-mlops `
  --env-from=configmap/k8s-mlops-config `
  --command -- python -m src.training.hpo_optuna

# 3. Réentraînement complet autonome en boucle fermée (Closed-Loop Retraining)
kubectl run mlops-retrain --rm -it --restart=Never --image=mlops-app:v1.0.0 -n k8s-mlops `
  --env-from=configmap/k8s-mlops-config `
  --command -- python -m src.training.retrain_pipeline

# 4. Déclenchement manuel immédiat des transformations analytiques dbt
kubectl create job --from=cronjob/k8s-mlops-dbt-runner dbt-manual-run -n k8s-mlops

# 5. Diagnostic de santé complet depuis l'intérieur du cluster
kubectl run mlops-check --rm -it --restart=Never --image=mlops-app:v1.0.0 -n k8s-mlops `
  --env-from=configmap/k8s-mlops-config `
  --command -- python -m src.utils.check_status

# 6. Grand Reset MLOps (Purge & Réinitialisation Contrôlée)
kubectl run mlops-reset --rm -it --restart=Never --image=mlops-app:v1.0.0 -n k8s-mlops `
  --env-from=configmap/k8s-mlops-config `
  --command -- python -m src.utils.full_reset_mlops
```

---

### Option B : Déclenchement via Webhook FastAPI (`curl`)
Déclenche de façon non bloquante la boucle de réentraînement via l'API REST sécurisée :

```bash
# Appel sécurisé au webhook de réentraînement
curl -X POST http://localhost:8000/webhooks/trigger-retrain \
  -H "Content-Type: application/json" \
  -H "x-webhook-secret: k8s-mlops-super-secure-token" \
  -d '{
    "source": "evidently-worker",
    "event_type": "drift_alert",
    "drift_score": 0.42,
    "details": {"metric": "ks_test_pvalue_drop"}
  }'
```

---

### Option C : Pilotage Interactif via le Portail Streamlit
Toutes les actions opérationnelles peuvent être déclenchées sans ligne de commande directement sur l'interface graphique :
* Rendez-vous sur **`http://localhost:8503`** (port-forward Streamlit).
* Page **`4_Webhooks`** : Simulez une dérive de données, auditez les logs d'événements et déclenchez le réentraînement en un clic.

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

