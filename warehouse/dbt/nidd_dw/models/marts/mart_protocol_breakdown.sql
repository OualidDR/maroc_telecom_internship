-- Protocol usage crossed with attack type.
-- Answers: "which protocols carry which attacks?" (e.g. confirms ICMPFlood
-- is exclusively icmp, UDPFlood is exclusively udp, scans favor tcp/udp).

with base as (
    select * from {{ ref('stg_flows') }}
)

select
    proto,
    attack_type,
    count(*) as flow_count,
    avg(totpkts) as avg_packets,
    avg(sMeanPktSz) as avg_src_packet_size
from base
group by proto, attack_type
order by proto, flow_count desc
