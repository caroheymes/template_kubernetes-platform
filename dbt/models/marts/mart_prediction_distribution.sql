-- mart_prediction_distribution.sql
-- Surveillance statistique de la distribution des sorties du modèle

with predictions as (
    select * from {{ ref('stg_predictions') }}
)

select
    received_date,
    model_version,
    count(prediction_id) as sample_count,
    round(min(predicted_value)::numeric, 4) as min_score,
    round(percentile_cont(0.25) within group (order by predicted_value)::numeric, 4) as q1_score,
    round(percentile_cont(0.50) within group (order by predicted_value)::numeric, 4) as median_score,
    round(percentile_cont(0.75) within group (order by predicted_value)::numeric, 4) as q3_score,
    round(max(predicted_value)::numeric, 4) as max_score,
    round(avg(predicted_value)::numeric, 4) as mean_score,
    round(stddev(predicted_value)::numeric, 4) as stddev_score
from predictions
group by received_date, model_version
order by received_date desc
