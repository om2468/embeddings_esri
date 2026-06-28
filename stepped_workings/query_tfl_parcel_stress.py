import argparse
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stepped_workings.analyze_tfl_par_land import load_env_file

METRICS = {
    "degraded": '"aef_18_25_pct_degraded"',
    "mean_dist": '"aef_18_25_mean_dist"',
    "hotspot_px": '"aef_18_25_hotspot_px"',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Query the most stressed TFL parcels within a borough.")
    parser.add_argument("--borough", required=True, help="Borough name, e.g. Brent")
    parser.add_argument("--metric", choices=sorted(METRICS.keys()), default="degraded")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def connect_postgres():
    env = load_env_file(ROOT / ".env")
    return psycopg.connect(
        host=env["PG_HOST"],
        port=env["PG_PORT"],
        dbname=env["PG_DATABASE"],
        user=env["PG_USERNAME"],
        password=env["PG_PASSWORD"],
    )


def main():
    args = parse_args()
    metric_sql = METRICS[args.metric]
    sql = f'''
        SELECT
            "OBJECTID",
            "PAR_ID",
            "COMPANY_DESC",
            "INTEREST_DESC",
            "aef_18_25_borough",
            "aef_18_25_px_count",
            ROUND(COALESCE("aef_18_25_mean_dist", 0)::numeric, 6) AS mean_dist,
            ROUND(COALESCE("aef_18_25_pct_degraded", 0)::numeric, 2) AS pct_degraded,
            "aef_18_25_hotspot_px",
            "aef_18_25_interp",
            "aef_18_25_dom_cluster"
        FROM tfl.tfl_par_land
        WHERE COALESCE("aef_18_25_borough", '') ILIKE %s
          AND COALESCE("aef_18_25_px_count", 0) > 0
        ORDER BY {metric_sql} DESC NULLS LAST, "aef_18_25_px_count" DESC, "OBJECTID" ASC
        LIMIT %s
    '''
    with connect_postgres() as conn, conn.cursor() as cur:
        cur.execute(sql, (args.borough, args.limit))
        rows = cur.fetchall()
    if not rows:
        print(f"No analysed parcels found for borough: {args.borough}")
        return
    print(f"Top {len(rows)} stressed parcels in {args.borough} by {args.metric}:")
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))


if __name__ == "__main__":
    main()