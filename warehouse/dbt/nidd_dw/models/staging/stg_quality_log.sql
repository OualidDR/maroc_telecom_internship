with source as (
    select * from {{ source('nidd_raw', 'quality_log') }}
)

select
    event_timestamp,
    batch_id,
    row_count,
    success,
    n_failed_expectations,
    failed_expectation_summary
from source
