{{ config(materialized='view') }} 
-- used view as user data is simple less compute and remains fresh

SELECT
    name as user_id,
    user_type,
    base_engagement,
    hourly_weights::jsonb as hourly_weights
FROM {{ source('public', 'raw_users') }}