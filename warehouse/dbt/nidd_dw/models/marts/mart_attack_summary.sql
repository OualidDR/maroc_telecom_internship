-- Volume of flows per attack type and tool, plus a share-of-total column.
-- Answers: "which attacks dominate the traffic, and which tool generates them?"

with base as (
    select * from {{ ref('stg_flows') }}
),

category as (
    select * from {{ ref('attack_type_category') }}
),

by_attack as (
    select
        base.attack_type,
        category.attack_category,
        base.attack_tool,
        base.label,
        count(*) as flow_count,
        sum(base.totbytes) as total_bytes,
        avg(base.rate) as avg_rate,
        avg(base.dur) as avg_duration_seconds
    from base
    left join category on base.attack_type = category.attack_type
    group by base.attack_type, category.attack_category, base.attack_tool, base.label
)

select
    *,
    round(100.0 * flow_count / sum(flow_count) over (), 2) as pct_of_total_flows
from by_attack
order by flow_count desc
