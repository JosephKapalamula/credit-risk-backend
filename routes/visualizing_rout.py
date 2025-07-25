from fastapi import APIRouter
from controllers.data_visilization import generate_visualizations 


visual_router = APIRouter()

@visual_router.get("/visualize", tags=["visualize"])
def get_visualizations():
    visualizations = generate_visualizations()
    return visualizations

