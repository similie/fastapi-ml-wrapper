from typing import Any
from datetime import datetime
from pytz import utc
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, UUID4

# Request and Response Classes for input/output validation #


# Request Classes
class BasePostRequest(BaseModel):
    '''
    Base POST Request model for generic post processing of model payloads
    by model name.
    '''
    model_config = ConfigDict(extra='allow')

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
        default_factory=lambda: int(datetime.now(utc).timestamp() * 1000)
    )


class BackgroundTaskResponse(BaseResponse):
    '''
    Response model returned to an API request that results in a background task
    being spawned. E.g. `/api/v1/predict`, the response returns immediately and
    queues a webhook reponse as a callback once the process has completed.
    '''
    status: int = 200
    message: str = ''


class DataTaskResponse(BackgroundTaskResponse):
    '''
    Response model returned to an inline (inference) call that shold return
    an array of data. Subclasses should override the list type.
    '''
    data: list[Any] | None = None


class WebhookResponse(DataTaskResponse):
    '''
    Response model POSTED to a callback webhook. Contains a unix millis timestamp
    from `BaseReponse`, a status code and message from `BackgroundTaskResponse`
    an optional for the data array from `dataTaskResponse`, if the event returns data
    and adds a property for the event name. The originating WebhookRequest->CallbackToken
    should be added to the response headers before executing the callback request.
    '''
    eventName: str


class ListTypeResponse(BaseResponse):
    '''
    A list of string values and count (with the current UTC unix millis from
    the superclass). E.g. All Prediction models available in this container
    '''
    count: int | None = 0
    data: list[str]


class TemplateResponse(BaseResponse):
    '''
    Container type for template responses from Prediction models. Note: uses
    "template" rather than "schema" to avoid prop name conflicts.
    '''
    model_config = ConfigDict(extra='forbid')  # schema should be fixed
    name: str
    version: str = '1'
    accepts: dict[str, Any]  # supply Json in your subclass
    events: list[str]
    returns: dict[str, Any]  # Json of BackgroundTaskResponse (subclass)
    notes: str = ''
