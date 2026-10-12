"""
config.py — Configuration centralisée pour k8s_mlops_platform
Charge les variables d'environnement et les Secrets Kubernetes.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    user: str = "mlops"
    password: str = "mlops_password"
    host: str = "localhost"
    port: int = 5432
    db_name: str = "mlops"

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    @property
    def url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    def get_url_for_db(self, db_name: str) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{db_name}"


class StorageSettings(BaseSettings):
    endpoint_url: str | None = None
    bucket_models: str = "models"
    bucket_data: str = "bronze"
    bucket_reports: str = "reports"
    access_key: str | None = None
    secret_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="S3_", extra="ignore")


class MLflowSettings(BaseSettings):
    tracking_uri: str = "http://localhost:5002"
    model_name: str = "k8s-mlops-xgboost"
    model_alias: str = "champion"
    experiment_name: str = "k8s-mlops-xgboost-experiment"

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", extra="ignore")


class WebhookSettings(BaseSettings):
    secret_token: str = "k8s-mlops-super-secure-token"
    notification_url: str | None = None

    model_config = SettingsConfigDict(env_prefix="WEBHOOK_", extra="ignore")


class AppSettings(BaseSettings):
    app_name: str = "k8s_mlops_platform"
    environment: str = "development"
    debug: bool = False
    serving_host: str = "0.0.0.0"
    serving_port: int = 8000

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")


db_settings = DatabaseSettings()
storage_settings = StorageSettings()
mlflow_settings = MLflowSettings()
webhook_settings = WebhookSettings()
app_settings = AppSettings()
