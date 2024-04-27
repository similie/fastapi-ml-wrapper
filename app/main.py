from fastapi import FastAPI
from ..src.controllers import ApiController

app = FastAPI(title='WebML-API', redoc_url=None)
app.include_router(ApiController.routes)
