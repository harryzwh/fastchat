import os
import uvicorn
import argparse
import uuid
import json

from types import MethodType
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastchat.utils import build_logger

from api_model_worker import ApiModelWorker

app = FastAPI()


def get_llm(model_name):
    if 'gemini' in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDGMxYJDddFGXlg_x3_RcxVpr-B3dqM7Eo"
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            convert_system_message_to_human=True,
        )

        def chunk_process(chunk):
            text = chunk.content
            error_code = 0
            finish_reason = chunk.response_metadata.get("finish_reason", None)
            return text, error_code, finish_reason

        return llm, chunk_process
    elif "qianfan" in model_name:
        from langchain_community.llms import QianfanLLMEndpoint
        API_Key = "HfrtUrTB7mgf7uhl8TO5IeRN"
        Secret_Key = "ERS0tzGu11TcJGAl6NFv5FZN6HFXUofH"
        os.environ["QIANFAN_AK"] = API_Key
        os.environ["QIANFAN_SK"] = Secret_Key
        os.environ["QIANFAN_BASE_URL"] = "https://aip.baidubce.com"
        llm = QianfanLLMEndpoint(endpoint="yi_34b_chat", streaming=True, )

        def chunk_process(chunk):
            text = chunk
            error_code = 0
            finish_reason = None
            return text, error_code, finish_reason

        return llm, chunk_process
    else:
        raise Exception('{} is not implemented'.format(model_name))


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await worker.acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = worker.create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await worker.acquire_worker_semaphore()
    output = worker.generate_gate(params)
    worker.release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return {
        "model_names": worker.model_names,
        "speed": 1,
        "queue_length": worker.get_queue_length(),
    }


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    prompt = params["prompt"]
    return {"count": len(str(prompt)), "error_code": 0}


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


def create_api_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21005)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21005")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    worker_id = str(uuid.uuid4())[:8]
    logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

    worker = ApiModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_names=['qianfan-api'],
        limit_worker_concurrency=args.limit_worker_concurrency,
        logger=logger,
        no_register=args.no_register,
        conv_template=None,
        context_len=16384,
        seed=args.seed,
    )

    llm, chunk_process = get_llm("qianfan-api")
    worker.llm = llm
    worker.chunk_process = chunk_process

    return args, worker


if __name__ == "__main__":
    args, worker = create_api_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
