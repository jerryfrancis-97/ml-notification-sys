{{ config(materialized="table")}}

WITH sends AS (
    SELECT event_id, user_id, send_timestamp FROM {{ source("public", "raw_sends") }}
),

responses AS (
    SELECT event_id, opened, response_delay_minutes FROM {{ source("public", "raw_responses") }}
)

SELECT
    s.event_id,
    s.user_id,
    s.send_timestamp,
    EXTRACT(HOUR FROM s.send_timestamp) as send_hour,
    TO_CHAR(s.send_timestamp, 'FMDay') as day_name,
    ( EXTRACT(DOW FROM s.send_timestamp) IN (0,6) ) as is_weekend,

    COALESCE(r.opened, 0) as is_opened,
    NULLIF(r.response_delay_minutes, -1) as delay_min

FROM sends s
LEFT JOIN responses r ON s.event_id = r.event_id




