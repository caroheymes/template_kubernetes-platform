---
title: "PRD — k8s_mlops_platform (Template MLOps Kubernetes Générique)"
status: "draft"
version: "1.2.0"
created: "2026-10-12"
updated: "2026-10-12"
author: "BMad Product Manager & System Architect"
---

# Product Requirements Document (PRD)
## k8s_mlops_platform — Plateforme MLOps Cloud-Native Kubernetes

> **Statut** : Spécification Fonctionnelle & Technique  
> **Dépôt Cible** : `k8s_mlops_platform` (Template public GitHub open-source)  
> **Modèle de Référence** : **XGBoost** (Tabulaire)  
> **Moteur HPO** : **Optuna** (Tuning bayésien d'hyperparamètres distribué)  
> **Explicabilité (XAI)** : **Shapash** (Interprétabilité locale et globale basée sur SHAP)  
> **Transformation Analytique** : **dbt** (`dbt-postgres` pour la modélisation des marts)  
> **Socle Données** : **PostgreSQL** (StatefulSet persistant multi-schémas)  
> **Dashboarding** : **Streamlit Multi-Pages** (`App`, `XAI`, `Analytics`, `Webhooks`)

---

## 1. Vision & Objectifs Produit

### 1.1 Problématique
La majorité des solutions MLOps présentent quatre limitations critiques pour un déploiement réel :
1. **Couplage fort à un domaine métier particulier** : code, schémas de données et modèles non transposables à un autre cas d'usage tabulaire.
2. **Boîte noire et déficit d'explicabilité** : absence d'outils XAI vulgarisés pour expliquer les décisions individuelles d'un modèle aux auditeurs et experts métier.
3. **Absence de modélisation analytique robuste** : requêtes SQL brutes dispersées, sans gouvernance, sans tests de données ni lignage (*data lineage*).
4. **Verrouillage propriétaire (Vendor Lock-in)** : adhérence à des services fermés (SageMaker, Vertex AI) au lieu d'une pile Kubernetes open-source et portable.

### 1.2 Vision Cible
Créer **`k8s_mlops_platform`**, un template GitHub clé en main pour déployer en quelques minutes une architecture MLOps complète, déclarative et 100 % open-source sur Kubernetes :
* **Modèle Champion** : **XGBoost** tabulaire optimisé par **Optuna**.
* **Explicabilité Native** : **Shapash** pour des explications visuelles compréhensibles.
* **Transformations & Vues Analytics** : **dbt** modélisant les tables et métriques dans PostgreSQL.
* **Serving REST & Webhooks** : **FastAPI** asynchrone ($< 50\text{ ms}$).
* **Interface Opérationnelle** : **Streamlit 4-Pages** (`App`, `XAI`, `Analytics`, `Webhooks`).
* **MLOps Lifecycle** : **MLflow** (Tracking & Registry) et **Evidently** (Drift & Auto-retraining).

---

## 2. Architecture des Composants & Flux de Données

```
                                  Ingress NGINX
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  FastAPI Serving               Streamlit Dashboard            MLflow Server
 (Inférence & Webhooks)        (Multi-Pages App / XAI)      (Tracking & Registry)
  Port: 8000 /docs             (Analytics & Webhooks)        Port: 5002
           │                     Port: 8503                          │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
             ┌───────────────────────────────────────────────────────┐
             │       PostgreSQL (StatefulSet K8s + Volume PVC)       │
             │   - Base `mlops` :                                    │
             │       • `features` (Features Store brut & historisé)  │
             │       • `predictions` (Logs d'inférence en direct)    │
             │       • `analytics` (Vues & Marts modélisés par dbt)  │
             │       • `webhooks` (Journal des événements & alertes) │
             │   - Base `mlflow` : Métadonnées expériences & runs    │
             │   - Base `optuna` : Synchronisation des études HPO    │
             └──────────────────────────┬────────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     Stockage Objet S3 / MinIO                       Calcul & Automatisation
   - Modèles XGBoost (.json/.ubj)                   - Optuna HPO Worker (Ray/K8s)
   - Objets Shapash SmartExplainer                  - Evidently Drift Detector Job
   - Rapports Evidently (HTML/JSON)                 - dbt Transform Job (K8s CronJob)
   - Artefacts de validation                        - Retraining & Promotion Pipeline
```

---

## 3. Rôle Central de PostgreSQL & Couches de Données dbt

PostgreSQL est déployé via un **StatefulSet Kubernetes** avec volume persistant (`PersistentVolumeClaim`). Le projet **dbt** structure les couches analytiques :

| Base de Données | Schéma SQL | Entité Responsable | Rôle & Description |
| :--- | :--- | :--- | :--- |
| **`mlops`** | `features` | Pipeline Ingestion | Jeux d'entraînement tabulaires, tables de features brutes et partitions temporelles. |
| **`mlops`** | `predictions` | FastAPI Serving | Historique brut de chaque requête reçue (`request_id`, features JSON, prédiction XGBoost, latence). |
| **`mlops`** | `analytics` | **dbt-postgres** | Tables et vues modélisées par dbt pour l'analytique : volume horaire, débits, distribution des scores, métriques de performance. |
| **`mlops`** | `webhooks` | FastAPI / Evidently | Journal des événements webhooks reçus et émis (alertes de dérive, triggers de réentraînement, notifications). |
| **`mlflow`** | `public` | MLflow Server | Backend store (expériences, hyperparamètres, métriques, artefacts, Model Registry). |
| **`optuna`** | `public` | Optuna Engine | RDB Storage partagé pour la synchronisation multi-trials et la reprise sur panne. |

---

## 4. Exigences Fonctionnelles Détaillées (Functional Requirements)

### 4.1 Initialisation & Déploiement K8s (FR-100 à FR-103)
* **FR-100 — Amorçage Cluster depuis Docker** : Commandes documentées et testées pour initialiser le cluster local soit via **Docker Desktop** (Kubernetes intégré), soit via **Minikube** avec le pilote Docker (`minikube start --driver=docker --cpus=4 --memory=8192`).
* **FR-101 — Déploiement Déclaratif Helm** : Déploiement complet en une commande unique :
  ```powershell
  helm upgrade --install k8s-mlops ./helm/ -n k8s-mlops --create-namespace -f ./helm/values.yaml
  ```
* **FR-102 — Gestion des Secrets** : Mots de passe PostgreSQL, clés d'accès S3/MinIO et tokens webhooks gérés par des `Secrets` K8s.
* **FR-103 — Zéro Référence Métier** : Aucun terme spécifique à un ancien cas d'usage ; variables et schémas 100 % génériques.

### 4.2 Inférence & Webhooks FastAPI (FR-200 à FR-205)
* **FR-200 — Inférence XGBoost Unitaire** (`POST /predict`) : Charge le modèle XGBoost Champion en mémoire, valide la charge utile JSON et retourne la prédiction en $< 50\text{ ms}$.
* **FR-201 — Inférence par Lots** (`POST /predict/batch`) : Inférence vectorisée pour les volumes élevés de données.
* **FR-202 — Endpoint d'Explicabilité à la Requête** (`POST /predict/explain`) : Retourne la contribution SHAP/Shapash des principales variables pour la requête envoyée.
* **FR-203 — Gestionnaire de Webhooks Entrants** (`POST /webhooks/trigger-retrain`) :
  * Sécurisé par token HMAC ou header `X-Webhook-Secret`.
  * Déclenche un Job Kubernetes de réentraînement lorsqu'une dérive est confirmée ou qu'un batch de nouvelles données arrive.
* **FR-204 — Émission de Webhooks Sortants** (`POST` vers Slack/Teams/Discord) : Envoi de notifications formatées lors des événements clés (modèle promu, dérive détectée, fin d'entraînement).
* **FR-205 — Sondes K8s & Métriques** : `/health/live`, `/health/ready` et `/metrics` (Prometheus).

### 4.3 Optimisation d'Hyperparamètres Optuna & XGBoost (FR-300 à FR-303)
* **FR-300 — Études HPO Distribuées Optuna** : Module `src/training/hpo_optuna.py` optimisant par méthode bayésienne (TPESampler) : `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `n_estimators`.
* **FR-301 — Stockage RDB Persistant** : Sauvegarde des études dans PostgreSQL (`postgresql://.../optuna`).
* **FR-302 — Intégration Native MLflow** : Chaque trial Optuna est journalisé comme un run imbriqué dans MLflow (`MLflowCallback`).
* **FR-303 — Sauvegarde de l'Explainer** : Entraîne et sérialise conjointement le modèle XGBoost et l'objet `SmartExplainer` Shapash.

### 4.4 Explicabilité XAI via Shapash (FR-400 à FR-403)
* **FR-400 — Objet SmartExplainer Centralisé** : Module `src/xai/explainer.py` encapsulant le modèle XGBoost, les libellés compréhensibles des features et les encodeurs.
* **FR-401 — Explicabilité Globale** : Calcul de l'importance globale des variables (Feature Importance Plot) et visualisations des interactions de features.
* **FR-402 — Explicabilité Locale** : Visualisation locale détaillée pour chaque prédiction (Waterfall Plot, contributions positives et négatives à la décision).
* **FR-403 — Export Web & Artefacts** : Sauvegarde des explications sous forme de graphiques HTML interactifs et export vers S3/MLflow.

### 4.5 Modélisation & Vues Analytiques via dbt (FR-500 à FR-503)
* **FR-500 — Projet dbt Structuré (`dbt/`)** : Projet dbt configuré avec `dbt-postgres` pointant sur la base `mlops`.
* **FR-501 — Couches de Modélisation dbt** :
  * `models/staging/` : Vues de nettoyage sur `features` et `predictions`.
  * `models/marts/` : Tables analytiques finales agrégées :
    * `mart_inference_kpis` : volume journalier, latence médiane et $p_{95}$, taux d'erreur HTTP.
    * `mart_prediction_distribution` : quantiles des prédictions, dérive empirique des scores.
* **FR-502 — Tests de Données dbt Automatisés** : Tests `unique`, `not_null`, et tests de bornes (`accepted_values`) sur les features et prédictions.
* **FR-503 — Exécution K8s CronJob** : Un `CronJob` Kubernetes exécute `dbt run && dbt test` toutes les heures pour actualiser les marts.

### 4.6 Dashboard Streamlit Multi-Pages (FR-600 à FR-604)
L'interface Streamlit est organisée en **4 pages distinctes** :

* **FR-601 — Page 1 : `App` (Inference & Simulation)** :
  * Formulaire interactif de saisie des features tabulaires.
  * Bouton « Prédire » appelant directement l'API FastAPI.
  * Affichage de la prédiction en direct avec jauge de confiance et historique récent.
* **FR-602 — Page 2 : `XAI` (Shapash Explainability)** :
  * Sélecteur de prédiction pour afficher l'explication locale immédiate (Waterfall plot Shapash).
  * Vue d'importance globale des variables du modèle en production.
  * Filtres par features pour analyser les dépendances partielles (*feature contributions*).
* **FR-603 — Page 3 : `Analytics` (dbt Marts)** :
  * Métriques de production branchées directement sur les tables dbt (`analytics.mart_*`).
  * Graphiques Plotly : volume de requêtes dans le temps, latence $p_{95}$, distribution des prédictions vs historique.
  * État de fraîcheur des modèles dbt et résultats des tests de validation de données.
* **FR-604 — Page 4 : `Webhooks` (Monitoring & Triggers)** :
  * Visualisation du journal des webhooks reçus et émis (dernières 24 heures).
  * Bouton d'action « Déclencher Réentraînement » émettant un webhook sécurisé vers le pipeline.
  * Interface de configuration des webhooks sortants (URLs de destination, canaux Slack/Discord).
  * Statut en direct de la dérive statistique Evidently (KS-test).

### 4.7 MLOps Lifecycle & Auto-Promotion (FR-700 à FR-703)
* **FR-700 — Tracking & Model Registry MLflow** : Enregistrement automatique des runs, paramètres, métriques et artefacts (modèle + explainer Shapash).
* **FR-701 — Surveillance de Dérive (Evidently)** : Détection de drift sur les tables dbt `staging`, génération de rapports HTML sur S3.
* **FR-702 — Promotion Automatique Champion vs Challenger** : Le nouveau modèle devient `champion` si sa métrique de validation est supérieure d'au moins 2 % au Champion en titre.
* **FR-703 — Rechargement à Chaud** : FastAPI et Streamlit rechargent le modèle `champion` sans interruption de service.

### 4.8 Dossier de Tests & Assurance Qualité (FR-800 à FR-803)
* **FR-800 — Tests Unitaires (`tests/unit/`)** : Test unitaire de la logique XGBoost, des schemas Pydantic et des fonctions utilitaires.
* **FR-801 — Tests d'Intégration (`tests/integration/`)** : Validation de la chaîne de connexion PostgreSQL, de l'abstraction S3 et des commandes dbt.
* **FR-802 — Tests d'API (`tests/api/`)** : `TestClient` FastAPI validant `/predict`, `/predict/explain`, `/webhooks/*` et `/health`.
* **FR-803 — Validation Helm & Lint** : Exécution de `helm lint` et `ruff check .` garantissant 0 avertissement.

---

## 5. Exigences Non-Fonctionnelles (NFRs)

| Identifiant | Domaine | Exigence Spécifique |
| :--- | :--- | :--- |
| **NFR-SEC-01** | Sécurité K8s | Conteneurs non-root (UID `10001`), `allowPrivilegeEscalation: false`, profil **Restricted** des Pod Security Standards. |
| **NFR-PERF-01** | Latence Inférence | Inférence unitaire XGBoost via FastAPI $< 50\text{ ms}$ au 95e percentile ($p_{95}$). |
| **NFR-REL-01** | Persistance Données | PostgreSQL sur volume persistant K8s (PVC) avec politique de rétention `Retain`. |
| **NFR-MOD-01** | Modularité | dbt, Shapash et Streamlit fonctionnent comme des composants découplés ; une panne sur Streamlit n'affecte en rien l'API FastAPI ni le tracking MLflow. |
| **NFR-PORT-01** | Portabilité | Fonctionnement garanti en local (Docker Desktop / Minikube) et sur cloud managé (EKS, GKE) sans modification de code. |

---

## 6. Structure Arborescente Complète de `k8s_mlops_platform`

```
k8s_mlops_platform/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint Ruff, Tests Pytest, Helm Lint, dbt test
│       └── cd.yml                 # Build Docker & Déploiement K8s
├── dbt/                           # Projet dbt pour les transformations analytiques
│   ├── dbt_project.yml            # Configuration du projet dbt
│   ├── profiles.yml               # Connexion PostgreSQL (injectée par env K8s)
│   └── models/
│       ├── staging/
│       │   ├── stg_features.sql
│       │   └── stg_predictions.sql
│       └── marts/
│           ├── mart_inference_kpis.sql
│           └── mart_prediction_distribution.sql
├── docker/
│   ├── Dockerfile.serving         # Image FastAPI Inférence & Webhooks (légère, non-root)
│   └── Dockerfile.app             # Image Streamlit 4-Pages & Workers
├── helm/
│   ├── Chart.yaml                 # Métadonnées du Chart Helm k8s-mlops
│   ├── values.yaml                # Configuration centralisée (ports, replicas, DB)
│   └── templates/
│       ├── configmaps.yaml        # Paramètres d'environnement partagés
│       ├── secrets.yaml           # Secrets chiffrés / externalisés
│       ├── postgres.yaml          # StatefulSet PostgreSQL + PVC 10Gi
│       ├── mlflow.yaml            # Deployment MLflow Tracking & Registry
│       ├── serving.yaml           # Deployment FastAPI Inférence & HPA
│       ├── streamlit.yaml         # Deployment Dashboard Streamlit Multi-Pages
│       ├── dbt-cronjob.yaml       # CronJob K8s exécutant dbt run & dbt test
│       ├── kuberay-cluster.yaml   # Ressource RayCluster pour calcul élastique
│       └── ingress.yaml           # Routage Ingress NGINX
├── src/
│   ├── common/
│   │   ├── config.py              # Configuration Pydantic Settings
│   │   ├── db.py                  # Usine SQLAlchemy PostgreSQL multi-bases
│   │   └── storage.py             # Client de stockage S3 / MinIO
│   ├── serving/
│   │   ├── main.py                # Application FastAPI (/predict, /webhooks)
│   │   ├── schemas.py             # Modèles de données Pydantic v2
│   │   ├── predictor.py           # Moteur d'inférence XGBoost
│   │   └── webhooks.py            # Logique de réception et émission de webhooks
│   ├── training/
│   │   ├── train_xgboost.py       # Script d'entraînement XGBoost
│   │   ├── hpo_optuna.py          # Étude d'optimisation Optuna distribuée
│   │   └── evaluate.py            # Logique Champion vs Challenger
│   ├── xai/
│   │   └── explainer.py           # Module Shapash SmartExplainer (local/global)
│   ├── tracking/
│   │   └── mlflow_client.py       # Helper de logging et promotion de modèles
│   ├── monitoring/
│   │   └── drift_detector.py      # Détection de dérive Evidently (KS-test)
│   └── dashboard/
│       ├── app.py                 # Point d'entrée Streamlit
│       └── pages/
│           ├── 1_App.py           # Page 1: Saisie & Inférence en direct
│           ├── 2_XAI.py           # Page 2: Explicabilité Shapash (local & global)
│           ├── 3_Analytics.py     # Page 3: Vues et KPIs analytiques dbt
│           └── 4_Webhooks.py      # Page 4: Journal et déclencheurs webhooks
├── tests/
│   ├── unit/                      # Tests unitaires purs (sans DB)
│   ├── integration/               # Tests avec mocks PostgreSQL et S3
│   ├── api/                       # Tests HTTP FastAPI TestClient
│   └── dbt/                       # Tests de syntaxe et compilation dbt
├── docs/
│   ├── prd.md                     # Ce document de spécification
│   ├── launch.md                  # Guide d'exécution et démarrage pas-à-pas
│   └── architecture.md            # Spécification technique détaillée
├── requirements.txt               # Dépendances Python socle
├── pyproject.toml                 # Configuration Ruff, Pytest, Coverage
├── .dockerignore
└── README.md                      # Présentation du template open-source
```

---

## 7. Critères d'Acceptation & Validation

1. [ ] **Cluster Initialisé** : Démarrage vérifié sur Docker Desktop K8s ou Minikube via `minikube start --driver=docker`.
2. [ ] **Bases PostgreSQL & dbt Opérationnels** : Schémas `features`, `predictions`, `analytics` et `webhooks` instanciés, exécution réussie de `dbt run && dbt test`.
3. [ ] **XGBoost + Optuna Validés** : Optimisation HPO exécutée avec journalisation automatique dans MLflow et modèle champion enregistré.
4. [ ] **Explicabilité Shapash Opérationnelle** : Génération des explications locales et globales consultables via l'API et la page dédiée Streamlit.
5. [ ] **FastAPI & Webhooks** : Endpoints `/predict`, `/predict/explain`, et `/webhooks/*` fonctionnels avec réponse en $< 50\text{ ms}$.
6. [ ] **Streamlit 4-Pages Actif** : Navigation fluide entre les 4 pages (`App`, `XAI`, `Analytics`, `Webhooks`) sans erreur de rafraîchissement.
7. [ ] **Suite de Tests Complète** : 100 % des tests unitaires, d'intégration, d'API et dbt passent au vert (`pytest tests/`).
