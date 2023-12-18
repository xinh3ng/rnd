from fastapi import APIRouter, Response, status
from pydantic import BaseModel


router = APIRouter(prefix="/health")


class APIHealth(BaseModel):
    database_is_online: bool = True
    saq_worker_is_online: bool = True


@router.get(
    "/",
    response_model=APIHealth,
    responses={503: {"description": "Some or all services are unavailable", "model": APIHealth}},
)
async def check_health(response: Response):
    """Check availability to get an idea of the api health."""
    # TODO: you can make this a bit more complete and detailed and add more checks
    #  like trying to insert a record in the db
    # logger.info("Health Check⛑")
    health = APIHealth()

    if not all(health.dict().values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health
