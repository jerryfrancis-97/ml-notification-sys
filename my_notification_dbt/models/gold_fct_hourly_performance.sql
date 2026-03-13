{{ config(materialized='table') }}

SELECT
    send_hour,
    COUNT(*) as total_sent,
    SUM(is_opened) as total_opened,
    ROUND(AVG(is_opened) * 100, 2) as open_rate_pct,
    ROUND(AVG(delay_min), 2) as avg_delay_min
FROM {{ ref('silver_stg-notifications') }}
GROUP BY send_hour
ORDER BY send_hour ASC