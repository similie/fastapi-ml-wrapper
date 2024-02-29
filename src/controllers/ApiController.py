from fastapi import APIRouter, Response  # , Request, BackgroundTasks

from app.config import getConfig
from ..prediction_models.AllModelsService import getModelNames, modelForPayload
# Request structs
from ..interfaces.ReqRes import BasePostRequest
# Response structs
from ..interfaces.ReqRes import BaseResponse, ListTypeResponse

# set up cached config
config = getConfig()

routes = APIRouter(
    prefix=config.apiEndpoint,
    tags=['model'],
    dependencies=None,
)

# TODO: Long running tasks, Predict, FineTune use Background tasks, with
# a webhook subscription in the payload body for receiving event messages
# See: https://fastapi.tiangolo.com/tutorial/background-tasks/


@routes.get('/')
async def home():
    '''
    Return a welcome message on API root endpoint
    '''
    return {"msg": "Welcome to the Web ML Wrapper"}


@routes.get('/models')
async def all_models() -> ListTypeResponse:
    '''
    Return the names of all models available in this container.
    '''
    data = getModelNames()
    return ListTypeResponse(data=data, count=len(data))


@routes.get('/template/{model_name}')
async def model_schema(model_name: str) -> BaseResponse:
    '''
    Returns the schema for the requested model or 404 if no model exists
    with the specfied name.
    '''
    model = modelForPayload(BasePostRequest(modelName=model_name))
    if model is not None:
        result = await model.template()
        return result

    return Response('Resource not found', 404)


# TODO: Endpoints for: inference, fine-tuning & training
@routes.post('/predict/{model_name}')
async def run_model_inference(model_name: str, req: BasePostRequest):
    '''
    Run inference on the specified model. This method returns immediately and
    runs the task in a background thread. To receive progress reports, you should
    also specify callback hooks in your post request.

    See the model template for details of the expected body format.
    '''
    model = modelForPayload(BasePostRequest(modelName=model_name))
    if model is not None:
        res = await model.predict(req)
        return res

    return Response('Resource not found', 404)


@routes.post('/fine-tune/{model_name}')
async def fine_tune_model(model_name: str, req: BasePostRequest):
    '''
    Trigger fine-tuning from a known data source. This method returns immediately
    and runs the task in a background thread. To receive progress reports, you
    should also specify callback hooks in your post request.

    See the model template for details of the expected body format.
    '''
    model = modelForPayload(BasePostRequest(modelName=model_name))
    if model is not None:
        res = await model.fineTune(req)
        return res

    return Response('Resource not found', 404)


@routes.post('/fine-tune-with-csv/{model_name}')
async def fine_tune_with_csv_upload(model_name: str, req: BasePostRequest):
    '''
    Upload a file (csv) and trigger fine-tuning. This method returns immediately
    and runs the task in a background thread. To receive progress reports, you
    should also specify callback hooks in your post request.

    See the model template for details of the expected body format.
    '''
    model = modelForPayload(BasePostRequest(modelName=model_name))
    if model is not None:
        res = await model.fineTune(req)
        return res

    return Response('Resource not found', 404)
