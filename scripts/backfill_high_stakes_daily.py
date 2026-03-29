from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from app.telemetry.high_stakes_backfill import backfill_high_stakes_daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill high-stakes daily aggregates from raw events")
    parser.add_argument("--days", type=int, default=365, help="How many days back from now to include")
    parser.add_argument("--start-ts", type=float, default=None, help="Optional start epoch seconds override")
    parser.add_argument("--end-ts", type=float, default=None, help="Optional end epoch seconds override")
    parser.add_argument("--max-events", type=int, default=500000, help="Maximum events to scan")
    args = parser.parse_args()

    end_ts = args.end_ts if args.end_ts is not None else datetime.now(timezone.utc).timestamp()
    start_ts = args.start_ts
    if start_ts is None:
        start_ts = (datetime.now(timezone.utc) - timedelta(days=max(1, args.days))).timestamp()

    result = backfill_high_stakes_daily(
        start_ts=start_ts,
        end_ts=end_ts,
        max_events=max(1, int(args.max_events)),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
