# ==============================================================================
# Dockerfile.app — Microservice Streamlit UI & dbt Client
# ==============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8503 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dépendances système pour PostgreSQL, curl (healthcheck) et git (dbt packages si besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (UID 10001) avec home directory (nécessaire pour cache ~/.streamlit et ~/.dbt)
RUN groupadd -r mlops && useradd -m -r -g mlops -u 10001 mlops

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie des modules applicatifs et du projet dbt avec propriétaire restreint
COPY --chown=mlops:mlops src/ /app/src/
COPY --chown=mlops:mlops dbt/ /app/dbt/

USER 10001

EXPOSE 8503

# Lancement du portail Streamlit (page d'accueil app.py)
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8503", "--server.address=0.0.0.0"]
