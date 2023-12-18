from fastapi import FastAPI

from rnd.apis.health import router as health_router
from rnd.ai.calendar.apis.routes.calendar_routes import router as calendar_router
from rnd.ai.iac.apis.routes.iac_routes import router as iac_router


def get_application() -> FastAPI:
    app = FastAPI(title="rnd api services", description="", debug=True)

    for r in [health_router, calendar_router, iac_router]:
        app.include_router(r)
    return app


app = get_application()
