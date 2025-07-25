from fastapi import APIRouter
from basemodel.model import InputData
from controllers.predict import predict 


predict_router = APIRouter()





@predict_router.post("/predict", tags=["predict"])
def input_predict(data: InputData):
    data=predict(data) 
    print(data)
    return data



