#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract offline WandB history records from a .wandb file.")
    parser.add_argument("--wandb-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def proto_items_to_dict(items):
    out = {}
    for item in items:
        try:
            out[item.key] = json.loads(item.value_json)
        except json.JSONDecodeError:
            out[item.key] = item.value_json
    return out


def main():
    args = parse_args()
    ds = datastore.DataStore()
    ds.open_for_scan(args.wandb_file)

    payload = {
        "wandb_file": args.wandb_file,
        "history": [],
        "summary": [],
        "run": None,
    }

    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        if kind == "history":
            payload["history"].append(proto_items_to_dict(rec.history.item))
        elif kind == "summary":
            payload["summary"].append(proto_items_to_dict(rec.summary.update))
        elif kind == "run":
            payload["run"] = {
                "run_id": rec.run.run_id,
                "project": rec.run.project,
                "display_name": rec.run.display_name,
                "job_type": rec.run.job_type,
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
