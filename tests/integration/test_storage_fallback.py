"""Tests d'intégration pour le client Object Storage avec fallback local."""

from src.common.config import storage_settings
from src.common.storage import ObjectStorageClient


def test_storage_settings_default_buckets():
    """Vérifie la nomenclature générique standardisée des buckets S3/MinIO."""
    assert storage_settings.bucket_data == "bronze"
    assert storage_settings.bucket_models == "models"
    assert storage_settings.bucket_reports == "reports"


def test_object_storage_local_fallback(tmp_path):
    """Vérifie l'écriture et la relecture locale de fichiers JSON sans S3/MinIO."""
    client = ObjectStorageClient()
    # Force le mode local en désactivant le client S3
    client._s3 = None

    test_bucket = "bronze"
    test_key = "models/metadata.json"
    payload = {"model_name": "xgboost-test", "version": "1.0.0", "accuracy": 0.945}

    client.put_json(test_bucket, test_key, payload)

    # Relecture
    recovered = client.get_json(test_bucket, test_key)
    assert recovered is not None
    assert recovered["model_name"] == "xgboost-test"
    assert recovered["accuracy"] == 0.945

    # Vérification d'une clé inexistante
    missing = client.get_json(test_bucket, "non_existent_file.json")
    assert missing is None
