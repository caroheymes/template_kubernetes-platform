-- stg_predictions.sql
-- Extraction et normalisation des requêtes d'inférence en direct

with source_predictions as (
    select
        id,
        request_id,
        received_at,
        features_payload,
        predicted_value,
        latency_ms,
        model_version
    from {{ source('mlops_raw', 'fact_predictions') }}
)

select
    id as prediction_id,
    request_id,
    received_at,
    date_trunc('hour', received_at) as received_hour,
    date_trunc('day', received_at) as received_date,
    predicted_value,
    latency_ms,
    model_version,
    features_payload
from source_predictions
