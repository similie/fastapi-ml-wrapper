from datetime import date, datetime
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, UUID4
#from app.models import MLModel List

## Request and Response Classes for input/output validation ##

# Request Classes
class BasePostRequest(BaseModel):
  '''
  Base POST Request model for generic post processing of model payloads
  by model name.
  '''
  model_config = ConfigDict(extra='ignore')
  
  modelName: str | None = None

class WebhookRequest(BasePostRequest):
  '''
  Struct encapsulating a webhook
  '''
  callbackUrl: HttpUrl
  eventName: str
  id: UUID4 = Field(
    default_factory=lambda: uuid4()
  )

# Response Classes
class BaseResponse(BaseModel):
  '''
  Basic response model with the current unix millis ['utc']
  at the point of sending the response
  '''
  model_config = ConfigDict(extra='allow')

  timestamp: int = Field(
    default_factory=lambda: int(datetime.now().timestamp() * 1000)
  )

class ListTypeResponse(BaseResponse):
  '''
  Returns a list of string values and count (with the current UTC unix millis
  from the superclass). E.g. All Prediction models available in this container
  '''
  count: int | None = 0
  data: list[str]

class TemplateRespose(BaseResponse):
  '''
  Container type for template responses from Prediction models. Note: uses
  "template" rather than "schema" to avoid prop name conflicts. 
  '''
  model_config = ConfigDict()
  template: dict[str, Any]
  # TODO: name, version, events, schema. E.g:
  # class BaseTemplate[T](BaseModel):
  #   '''
  #   The Filter Template describes the general schema for a class
  #   '''
  #   name: str
  #   version: str
  #   schema: T
  #   events: EventList | None = None
  #   returns: FilterResponse
