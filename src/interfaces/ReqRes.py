from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, UUID4

# Request and Response Classes for input/output validation #


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
    Struct encapsulating a webhook. When the specified event triggers, POSTS
    a JSON encoded WebhookResponse to the callback url, adding the Auth token
    to the response header in the form: Authorization: bearer TOKEN. Responses
    from this callback are discarded.
    '''
    callbackUrl: HttpUrl
    callbackAuthToken: UUID4
    eventNames: list[str] = '*'
    id: UUID4 = Field(
        default_factory=lambda: uuid4()
    )


# Response Classes
class BaseResponse(BaseModel):
    '''
    Base response model. Instantiated with the current unix millis UTC
    '''
    model_config = ConfigDict(extra='allow')

    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class WebhookResponse(BaseResponse):
    '''
    Response model POSTED to a callback webhook. Contains a unix millis
    timestamp (from BaseReponse) status code, event name and message
    '''
    status: int = 200
    eventName: str = '*'
    message: str = ''


class ListTypeResponse(BaseResponse):
    '''
    A list of string values and count (with the current UTC unix millis from
    the superclass). E.g. All Prediction models available in this container
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
