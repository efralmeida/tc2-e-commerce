from __future__ import annotations

import argparse
import json

from tc2_ecommerce.api import health
from tc2_ecommerce.app import App


def main() -> None:
    parser = argparse.ArgumentParser(description="TC2 E-commerce entrypoint")
    parser.add_argument("--action", choices=["health", "train", "evaluate"], default="health")
    parser.add_argument("--model", default="dummy")
    args = parser.parse_args()

    app = App()

    if args.action == "health":
        print(json.dumps(health(), ensure_ascii=True))
        return

    if args.action == "train":
        result = app.train(args.model)
        print(json.dumps(result["metrics"], ensure_ascii=True))
        return

    if args.action == "evaluate":
        metrics = app.evaluate(args.model)
        print(json.dumps(metrics, ensure_ascii=True))
        return


if __name__ == "__main__":
    main()