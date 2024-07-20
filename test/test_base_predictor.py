import os
import signal
from multiprocessing import Process
from logging import basicConfig, info
from typing import Any, Annotated
from json import loads
from httpx import Response, get
import pytest
from fastapi import FastAPI, Header, status, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as uvicornRun
from time import sleep

from src.interfaces.ReqRes import (
    BasePostRequest,
    WebhookRequest,
    WebhookResponse
)
from src.prediction_models.TestPredictor import ATestPredictor


def MockPostRequest() -> BasePostRequest:
    payload = {'modelName': 'test', 'device': '123ABC', 'numeric': 1.234}
    result = BasePostRequest.model_validate(payload)
    return result


def MockTestPredictor() -> ATestPredictor:
    req = MockPostRequest()
    return ATestPredictor(req)


def test_is_class_initialised():
    '''
    This tests the creation of a [xxx]PredictorClass from a payload
    containing a model_name key:value pair
    '''
    req = MockPostRequest()
    predictor = ATestPredictor(req)
    assert predictor is not None
    assert predictor.payload.modelName == 'test'
    assert predictor.__class__.__name__ == 'ATestPredictor'
    del predictor


def test_mock_instance_creator():
    '''
    Test our MockPredictor is creating the correct class instance.
    '''
    predictor = MockTestPredictor()
    assert predictor is not None


@pytest.mark.asyncio
async def test_predictor_template():
    predictor = MockTestPredictor()
    t = await predictor.template()
    assert t is not None
    assert t.name == 'ATestPredictor'
    assert t.notes == 'test predictor schema'


@pytest.mark.asyncio
async def test_predictor_predict():
    predictor = MockTestPredictor()
    predictionPayload = {'f1': 0, 'f2': 1}
    result: dict[str, Any] = await predictor.predict(predictionPayload)
    assert result is not None
    assert result['count'] == 0
    assert result['payload']['f1'] is not None


# TODO when method is available
# @pytest.mark.asyncio
# async def test_predictor_finetune():
#     predictor = MockTestPredictor()
#     predictionPayload = {'f1': 0, 'f2': 1}
#     result: dict[str, Any] = await predictor.fineTune(predictionPayload)
#     assert result is not None
#     assert result['count'] == 0
#     assert result['payload']['f1'] is not None


# TODO when method is available
# @pytest.mark.asyncio
# async def test_predictor_train():
#     predictor = MockTestPredictor()
#     predictionPayload = {'f1': 0, 'f2': 1}
#     result: dict[str, Any] = await predictor.train(predictionPayload)
#     assert result is not None
#     assert result['count'] == 0
#     assert result['payload']['f1'] is not None


def test_add_webhook():
    predictor = MockTestPredictor()
    hook = WebhookRequest(
        modelName='test',
        callbackUrl='http://example.com/callback',
        callbackAuthToken='e572b49a-f075-4c74-9be7-f3b5eb7ed33c',
        eventNames=['onTest']
    )
    assert hook.id is not None
    assert predictor.webhooks is not None
    assert len(predictor.webhooks) == 0

    predictor.setWebhook(hook)
    assert len(predictor.webhooks) == 1


# Set up a webhook response server and webook request & response instances.
# We use a FastAPI server running in a separate process to receive the test
# class' calls to POST->Webhook response. Tests are for internal status codes
# returned for different pathways and a 200 response from the remote server for
# the case that everything went as expected. We `kill` the remote server at the
# end of the test. The routing functions will need to know about the data they
# should be processing ~ they would be the systems setting up and sending the
# calls with Webhooks in them. They are declared next along with the server.

class webhookFixture():
    req = WebhookRequest(
        modelName='test',
        callbackUrl=f'http://127.0.0.1:8088/test',
        callbackAuthToken='e572b49a-f075-4c74-9be7-f3b5eb7ed33c',
        eventNames=['onTest']
    )
    res = WebhookResponse(
        message='web hook sent',
        data=[1,2,3,4,5],
        eventName='onTest'
    )


fixture = webhookFixture()
origins = [
    "http://localhost:8088",
    "http://127.0.0.1:8088",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/test')
async def endpointForWebhook(req: Request) -> PlainTextResponse:
    '''
    Endpoint template for the destination server that receives webhooks. We
    should ensure that the header: `X-Webhook-Token` contains the correct UUID
    and return a 20x status code. Otherwise we send back a suitable error code.
    '''
    msg = 'Thanks for calling'
    authToken = req.headers.get('x-webhook-token')

    # 1. No auth token, Bad Request
    if authToken is None:
        return PlainTextResponse(content=msg, status_code=400)

    # Case 2. Incorrect auth token, Not authorised
    expectedToken = str(fixture.req.callbackAuthToken)
    if authToken != expectedToken:
        return PlainTextResponse(content=msg, status_code=401)

    # Case 3. Incorrect event name, not acceptable (or precondition failed?)
    body = loads(await req.body())
    hook: WebhookResponse = WebhookResponse.model_validate_json(body)
    expectedEventName = fixture.res.eventName
    if hook.eventName != expectedEventName:
        # status.HTTP_406_NOT_ACCEPTABLE
        # or ?
        # status.HTTP_412_PRECONDITION_FAILED
        return PlainTextResponse(content=msg, status_code=412)

    # Case 4. No errors, OK
    return PlainTextResponse(content=msg, status_code=200)


@app.get('/test')
def getEndPointForWebhook():
    return PlainTextResponse(content='Thanks for calling', status_code=201)
# app.add_api_route(path='/test', endpoint=endpointForWebhook, methods=['GET', 'POST'])

# Global process container for FastAPI webserver
_process = None

def runServer():
    '''Allocation and run the server'''
    uvicornRun(app=app, host='localhost', port=8088)

def startAPIServer():
    '''Start the web server assigning the process class'''
    global _process
    _process = Process(target=runServer, args=(), daemon=True)
    _process.start()

@pytest.mark.asyncio
async def test_call_webhook_if_needed():
    basicConfig()
    predictor = MockTestPredictor()

    try:
        startAPIServer()
        sleep(1)

        # 1. Ensure the web service is running
        res = get('http://127.0.0.1:8088/test')
        assert res.status_code == 201
        assert 'Thanks for calling' in res.text

        statusCode = await predictor.sendWebhookIfNeeded(fixture.res)
        assert statusCode == 202  # http202-accepted, no webhooks added

        predictor.setWebhook(fixture.req)
        statusCode = await predictor.sendWebhookIfNeeded(fixture.res)
        assert statusCode == 200

        # TODO:
        # incorrect token
        # missing url in webhook

    finally:
        os.kill(_process.pid, signal.SIGTERM)
        sleep(0.5)
        assert _process.is_alive() is False


#     mockResponse = Response(
#         status_code=200,
#         headers=[{'X-Webhook-Token', hookReq.callbackAuthToken.hex}],
#         json=hookRes.model_dump_json()
#     )

#     predictor.setWebhook(hookReq)
#     apiRoute = respx.post(baseUrl).mock(return_value=mockResponse)
#     res = await predictor.sendWebhook(hookReq, hookRes)

#     assert res is not None
#     print(res)


# ~~~ This works ~~~
# @pytest.mark.asyncio
# async def test_mock_fastapi():
#     basicConfig()

#     try:
#         startAPIServer()
#         info(f'1. PID:{_process.pid}, is alive:{_process.is_alive()}, exit code:{_process.exitcode}')
#         info('sleep 2s')
#         sleep(2)    # or find a way to wait until the server is responding

#         info('continue')
#         info(f'2. PID:{_process.pid}, is alive:{_process.is_alive()}, exit code:{_process.exitcode}')

#         res = get('http://127.0.0.1:8088/test')

#         assert res.status_code == 201
#         assert 'Thanks for calling' in res.text

#     finally:
#         os.kill(_process.pid, signal.SIGTERM)
#         sleep(0.5)
#         info(f'3. PID:{_process.pid}, is alive:{_process.is_alive()}, exit code:{_process.exitcode}')

#         assert _process.is_alive() is False

    
