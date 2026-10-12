-- mart_inference_kpis.sql
-- Agrégations temporelles pour le monitoring de performance opérationnelle

with predictions as (
    select * from {{ ref('stg_predictions') }}
)

select
    received_date,
    model_version,
    count(prediction_id) as total_requests,
    round(avg(latency_ms)::numeric, 2) as latency_avg_ms,
    round(percentile_cont(0.50) within group (order by latency_ms)::numeric, 2) as latency_p50_ms,
    round(percentile_cont(0.95) within group (order by latency_ms)::numeric, 2) as latency_p95_ms,
    round(percentile_cont(0.99) within group (order by latency_ms)::numeric, 2) as latency_p99_ms,
    min(received_at) as first_request_at,
    max(received_at) as last_request_at
from predictions
group by received_date, model_version
order by received_date desc
