{{ config(materialized='table') }}

WITH notifications AS (
    SELECT * FROM {{ ref('silver_stg-notifications') }}
),

users AS (
    SELECT * FROM {{ ref('silver_stg-users') }}
)

SELECT
    u.user_type,
    COUNT(n.event_id) AS total_notifications_sent,
    SUM(n.is_opened) AS total_opens,
    ROUND(AVG(n.is_opened) * 100, 2) AS open_rate_pct,
    ROUND(AVG(n.delay_min), 2) AS avg_delay_min
FROM notifications AS n
JOIN users AS u ON n.user_id = u.user_id
GROUP BY u.user_type
ORDER BY open_rate_pct DESC