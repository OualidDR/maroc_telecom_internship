-- Flow volume bucketed by hour, split by label, attack_type, and DoS/Recon category.
-- Answers: "how does traffic composition evolve over time, at each granularity
-- Power BI Page 1/2 need?" (SOC overview needs label-level, Attack Analysis
-- needs attack_type-level and category-level).
-- Note: ingestion_timestamp reflects when the simulator SENT the event, not
-- a real capture time (see rapport_contrat_donnees_5G-NIDD.md -- no real
-- chronological order is recoverable from the source CSV).

with base as (
    select * from {{ ref('stg_flows') }}
),

category as (
    select * from {{ ref('attack_type_category') }}
)

select
    date_trunc('hour', base.ingestion_timestamp) as flow_hour,
    base.label,
    base.attack_type,
    category.attack_category,
    count(*) as flow_count,
    sum(base.totbytes) as total_bytes
from base
left join category on base.attack_type = category.attack_type
group by flow_hour, base.label, base.attack_type, category.attack_category
order by flow_hour, base.label, base.attack_type
