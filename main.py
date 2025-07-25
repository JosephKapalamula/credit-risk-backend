from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import predict_router
from routes.visualizing_rout import visual_router
from routes.fairness_route import fairness_router


app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(predict_router, prefix="/api/v1")
app.include_router(visual_router, prefix="/api/v1")
app.include_router(fairness_router, prefix="/api/v1")

