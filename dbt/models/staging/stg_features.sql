-- stg_features.sql
-- Vue standardisée sur les données de features brutes de référence

with source_features as (
    select
        id,
        created_at,
        feature_name,
        feature_value,
        split_group
    from {{ source('mlops_raw', 'reference_features') }}
)

select
    id as record_id,
    created_at,
    feature_name,
    feature_value,
    coalesce(split_group, 'train') as split_group
from source_features
