"""
storage.py — Abstraction de stockage objet compatible S3 / MinIO avec fallback local
"""

import json
import logging
import os
from typing import Any

from .config import storage_settings

logger = logging.getLogger("k8s_mlops.storage")


class ObjectStorageClient:
    """Client de stockage objet S3/MinIO avec repli local pour les tests et le dev sans cloud."""

    def __init__(self):
        self._s3 = None
        if storage_settings.access_key and storage_settings.secret_key:
            try:
                import boto3

                self._s3 = boto3.client(
                    "s3",
                    endpoint_url=storage_settings.endpoint_url,
                    aws_access_key_id=storage_settings.access_key,
                    aws_secret_access_key=storage_settings.secret_key,
                )
                logger.info("Connecteur S3/MinIO initialisé.")
            except ImportError:
                logger.warning("boto3 non disponible, bascule sur le stockage disque local.")

    def put_json(self, bucket: str, key: str, data: dict[str, Any]) -> None:
        """Sérialise et écrit un dictionnaire JSON dans le bucket S3 ou en local."""
        serialized = json.dumps(data, default=str)
        if self._s3:
            self._s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=serialized.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            path = os.path.join("/tmp", "k8s_mlops_storage", bucket, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(serialized)

    def get_json(self, bucket: str, key: str) -> dict[str, Any] | None:
        """Lit un document JSON depuis S3 ou le disque local."""
        if self._s3:
            try:
                resp = self._s3.get_object(Bucket=bucket, Key=key)
                return json.loads(resp["Body"].read().decode("utf-8"))
            except Exception as e:
                logger.error(f"Erreur de lecture S3 ({bucket}/{key}) : {e}")
                return None
        else:
            path = os.path.join("/tmp", "k8s_mlops_storage", bucket, key)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            return None


storage_client = ObjectStorageClient()
