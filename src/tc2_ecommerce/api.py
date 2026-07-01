"""API layer for local serving and AWS Lambda integration."""

from __future__ import annotations

import json
from typing import Any

from tc2_ecommerce.app import App
from tc2_ecommerce.logging_utils import get_logger


logger = get_logger(__name__)
service = App()


def health() -> dict[str, str]:
  return {"status": "ok"}


def recommend(model_name: str, user_ids: list[int], k: int | None = None) -> dict[str, Any]:
  predictions = service.predict(model_name=model_name, user_ids=user_ids, k=k)
  return {
    "model_name": model_name,
    "k": k,
    "predictions": predictions,
  }


def evaluate(model_name: str) -> dict[str, Any]:
  metrics = service.evaluate(model_name)
  return {
    "model_name": model_name,
    "metrics": metrics,
  }


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
  """AWS Lambda entrypoint compatible with API Gateway proxy integration."""
  try:
    route = event.get("rawPath") or event.get("path") or "/"
    method = (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "GET").upper()
    body_raw = event.get("body")
    body = json.loads(body_raw) if isinstance(body_raw, str) and body_raw else {}

    if route.endswith("/health") and method == "GET":
      payload = health()
      return {"statusCode": 200, "body": json.dumps(payload)}

    if route.endswith("/recommend") and method == "POST":
      payload = recommend(
        model_name=str(body["model_name"]),
        user_ids=[int(u) for u in body["user_ids"]],
        k=int(body.get("k", 10)),
      )
      return {"statusCode": 200, "body": json.dumps(payload)}

    if route.endswith("/evaluate") and method == "POST":
      payload = evaluate(model_name=str(body["model_name"]))
      return {"statusCode": 200, "body": json.dumps(payload)}

    return {
      "statusCode": 404,
      "body": json.dumps({"error": "Route not found", "route": route, "method": method}),
    }
  except Exception as exc:  # pragma: no cover
    logger.exception("Erro na lambda_handler")
    return {
      "statusCode": 500,
      "body": json.dumps({"error": str(exc)}),
    }


def create_fastapi_app() -> Any:
  """Create FastAPI app when dependency is installed."""
  try:
    from fastapi import FastAPI
  except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
      "FastAPI não instalado. Execute: poetry add --group dev fastapi uvicorn"
    ) from exc

  app = FastAPI(title="TC2 E-commerce Recommender API")

  @app.get("/health")
  def _health() -> dict[str, str]:
    return health()

  @app.post("/recommend")
  def _recommend(payload: dict[str, Any]) -> dict[str, Any]:
    return recommend(
      model_name=str(payload["model_name"]),
      user_ids=[int(u) for u in payload["user_ids"]],
      k=int(payload.get("k", 10)),
    )

  @app.post("/evaluate")
  def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    return evaluate(model_name=str(payload["model_name"]))

  return app
