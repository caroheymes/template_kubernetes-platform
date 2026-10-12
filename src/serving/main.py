"""
main.py — Application FastAPI asynchrone pour le Serving et les Webhooks
"""

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from ..common.config import app_settings
from ..common.db import dispose_all_engines, get_engine
from ..xai.explainer import explainer_manager
from .predictor import predictor
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    WebhookResponse,
    WebhookTriggerRequest,
)
from .webhooks import webhook_manager

# Métriques Prometheus
PREDICTION_COUNTER = Counter("mlops_predictions_total", "Nombre total de prédictions servies")
PREDICTION_LATENCY = Histogram("mlops_prediction_latency_seconds", "Latence d'inférence en secondes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage et nettoyage à l'arrêt."""
    predictor.load_model()
    yield
    dispose_all_engines()


app = FastAPI(
    title="k8s_mlops_platform — Serving & Inference API",
    version="1.0.0",
    description="Microservice d'inférence XGBoost, explicabilité XAI Shapash et gestionnaire de webhooks",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



def verify_webhook_header(x_webhook_secret: str | None = Header(None)) -> None:
    """Dépendance de sécurité pour vérifier le token secret du webhook."""
    if not webhook_manager.verify_token(x_webhook_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token webhook secret invalide ou manquant",
        )


@app.get("/health/live", tags=["Health"])
def health_live():
    """Sonde liveness pour Kubernetes."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
def health_ready():
    """Sonde readiness pour Kubernetes : valide la disponibilité du modèle et de PostgreSQL."""
    db_ok = False
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1;")
            db_ok = True
    except Exception:
        db_ok = False

    is_ready = predictor.is_ready and db_ok
    if not is_ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="not_ready",
                model_loaded=predictor.is_ready,
                database_connected=db_ok,
            ).model_dump(),
        )

    return HealthResponse(
        status="ready",
        model_loaded=predictor.is_ready,
        database_connected=db_ok,
    )



@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(req: PredictionRequest):
    """Inférence unitaire XGBoost en temps réel (< 50ms)."""
    with PREDICTION_LATENCY.time():
        resp = predictor.predict(req)
        PREDICTION_COUNTER.inc()
        return resp


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(req: BatchPredictionRequest):
    """Inférence par lots vectorisée pour les volumes élevés."""
    start = time.perf_counter()
    results = [predictor.predict(item) for item in req.items]
    total_time = (time.perf_counter() - start) * 1000.0

    return BatchPredictionResponse(
        predictions=results,
        total_processed=len(results),
        total_latency_ms=round(total_time, 2),
    )


@app.post("/predict/explain", response_model=ExplanationResponse, tags=["XAI"])
def explain_prediction(req: PredictionRequest):
    """Inférence combinée à l'explicabilité locale (contributions SHAP/Shapash)."""
    pred_resp = predictor.predict(req)

    # Calcul explicabilité locale Shapash (avec fallback structuré)
    explanation = explainer_manager.explain_single_prediction(req.features)
    contributions = explanation.get("contributions", {})

    return ExplanationResponse(
        request_id=pred_resp.request_id,
        prediction=pred_resp.prediction,
        model_name=pred_resp.model_name,
        model_version=pred_resp.model_version,
        latency_ms=pred_resp.latency_ms,
        feature_contributions=contributions,
        top_positive_features=[k for k, v in contributions.items() if v > 0],
        top_negative_features=[k for k, v in contributions.items() if v <= 0],
    )


@app.post(
    "/webhooks/retrain",
    response_model=WebhookResponse,
    dependencies=[Depends(verify_webhook_header)],
    tags=["Webhooks"],
)
@app.post(
    "/webhooks/trigger-retrain",
    response_model=WebhookResponse,
    dependencies=[Depends(verify_webhook_header)],
    tags=["Webhooks"],
)
def trigger_retrain(event: WebhookTriggerRequest):
    """Webhook sécurisé déclenchant un réentraînement suite à une dérive ou arrivée de données."""
    return webhook_manager.handle_incoming_event(event)



@app.get("/metrics", tags=["Observability"])
def metrics():
    """Endpoint de métriques au format standard Prometheus."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "k8s_mlops_platform.src.serving.main:app",
        host=app_settings.serving_host,
        port=app_settings.serving_port,
        reload=app_settings.debug,
    )
