# The PredictionService creates a consistent interface between the models and
# the APIController. It provides a list of availalable models and a class
# factory function for creating Prediction class instances from a named model.
from ..interfaces.ReqRes import BasePostRequest

# IMPORTANT. Import all Classes that are referenced by _allModels below
# even if they are not directly used. The references need to be in scope
# or the factory function modelForPayload will fail to resolve.
from .BasePredictor import BasePredictor
from .RainfallPredictor import RainfallPredictor
from .TestPredictor import TestPredictor
# from MyPredictor[.py] import MyPredictor [Class]

# "_allModels" is a list of all the callable models in this container.
# Keys are sent back to the discovery service for use as model names for
# further API calls. As such a Value for each Key must resolve to a Prediction
# Class so that the 'modelForPayload' factory function can instantiate the
# correct class.
# 
# Notes.
# The 'modelForPayload' function heavily validates the input payload and
# returns a new instance of the specified class when succesful or None
# otherwise. It fails for the following validation cases:
# - if the payload doesn't contain a 'modelName' key
# - the key doesn't exist in _allModels,
# - the value associated with the key doesn't resolve to a known class,
# - that class isn't a subclass of BasePredictor (for type narrowing)
_allModels = {
  'rainfall': 'RainfallPredictor',
  'test': 'TestPredictor'
  }

# Return the list of available models
def getModelNames():
  return list(_allModels.keys())

# Factory function to instantiate a predictor class from payload.model
# See notes on _allModels above. 
def modelForPayload(payload: BasePostRequest):
  model = None
  modelName = payload.modelName
  if (any(modelName in x for x in _allModels)):
    className = _allModels[modelName]
    classRef = globals().get(className)
    if (classRef != None):
      classInstance = classRef(payload)
      if (isinstance(classInstance, BasePredictor)):
        model = classInstance

  return model
