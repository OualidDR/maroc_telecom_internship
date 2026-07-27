-- Data quality pass rate over time, bucketed by hour.
-- Answers: "what fraction of micro-batches passed Great Expectations validation,
-- and how many rows/expectations failed?" (Power BI Page 4: Data Quality)

with base as (
    select * from {{ ref('stg_quality_log') }}
)

select
    date_trunc('hour', event_timestamp) as batch_hour,
    count(*) as total_batches,
    sum(case when success then 1 else 0 end) as passed_batches,
    sum(case when success then 0 else 1 end) as failed_batches,
    round(100.0 * sum(case when success then 1 else 0 end) / count(*), 2) as pass_rate_pct,
    sum(row_count) as total_rows_validated,
    sum(n_failed_expectations) as total_failed_expectations
from base
group by batch_hour
order by batch_hour
