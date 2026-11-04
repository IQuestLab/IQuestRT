"""Rank-zero HTTP routes; no model work runs on the ASGI event loop."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from .messages import register_messages_routes
from .limits import HTTPBounds, is_loopback


def create_app(driver, host):
    app = FastAPI(title="IQuestRT", docs_url=None, redoc_url=None)
    app.add_middleware(HTTPBounds, loopback_only=is_loopback(host))
    register_messages_routes(app, driver)

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok" if driver.healthy else "unavailable"},
                            status_code=200 if driver.healthy else 503)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": driver.model_name, "object": "model", "owned_by": "iquestrt"}],
            "has_more": False, "first_id": driver.model_name, "last_id": driver.model_name}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return f"iquestrt:max_model_len {driver.max_model_len}\n" + driver.metrics.render()

    @app.get("/stats")
    def stats():
        return {"status": "ok" if driver.healthy else "unavailable",
                "running": int(driver.active is not None), "waiting": driver.jobs.qsize(),
                **driver.metrics.snapshot()}

    return app
