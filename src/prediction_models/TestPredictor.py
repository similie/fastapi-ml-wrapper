from typing import Any
from .BasePredictor import BasePredictor

class TestPredictor(BasePredictor):
  '''
  Implementation class of abstract BasePredictor for testing
  '''
  
  # Add a private var that points to FastAPI router object
  # expose the Instantiated router via a method/property.
  # in app start, cycle through all the MLModels, try to instantiate
  # an empty class and fetch the router property, if sucessful, 
  # add it as a sub-route to the app.router.
  # https://fastapi.tiangolo.com/reference/apirouter/
  # 
  # Let's switch the routes currently in ApiController so that they are: 
  # api/v1/model_name/function and then each predictor class can define 
  # it's own routes, mapping them to the relevant structure.
  # Keep all models and subscribe but use custom routes for predict, fine-tune and train. 

  async def template():
    return {'schema': 'test predictor schema'} 
  
  async def fineTune(self, payload: Any):
    return super().fineTune(payload)
  
  async def predict(self, payload: Any):
    return await super().predict(payload)
  
  async def train(self, payload: Any):
    return super().train(payload)

