from fastapi import APIRouter
from controllers.fairness import calculate_fairness_metrics


fairness_router = APIRouter()
@fairness_router.get("/fairness", tags=["fairness"])
def get_fairness_metrics():
    metrics = calculate_fairness_metrics()
    return metrics

