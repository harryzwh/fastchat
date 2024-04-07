import os
import uvicorn
import argparse
import uuid
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastchat.utils import build_logger

from api_model_worker import ApiModelWorker

app = FastAPI()


def get_llm(register_api_endpoint_file):
    api_endpoint_info = json.load(open(register_api_endpoint_file))
    for k, v in api_endpoint_info['env'].items():
        os.environ[k] = v

    if 'gemini' in api_endpoint_info['api_type']:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDGMxYJDddFGXlg_x3_RcxVpr-B3dqM7Eo"
        llm = ChatGoogleGenerativeAI(
            model=api_endpoint_info['model'],
            convert_system_message_to_human=True,
        )

        def chunk_process(chunk):
            text = chunk.content
            error_code = 0
            finish_reason = chunk.response_metadata.get("finish_reason", None)
            return text, error_code, finish_reason

        def error_process(error):
            text = error
            error_code = None
            req_id = None
            return text, error_code, req_id

        return llm, chunk_process, error_process
    elif "qianfan" in api_endpoint_info['api_type']:
        from langchain_community.llms import QianfanLLMEndpoint
        llm = QianfanLLMEndpoint(endpoint=api_endpoint_info['model'], streaming=True, )

        def chunk_process(chunk):
            text = chunk
            error_code = 0
            finish_reason = None
            return text, error_code, finish_reason

        def error_process(error):
            text = error.error_msg
            error_code = error.error_code
            req_id = error.req_id
            return text, error_code, req_id

        return llm, chunk_process, error_process
    else:
        raise Exception('API for {} is not implemented'.format(api_endpoint_info['api_type']))


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
    parser.add_argument("--model-name", type=str, required=True)
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
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
        default=r".\api_endpoints.json",
        # required=True
    )
    args = parser.parse_args()
    worker_id = str(uuid.uuid4())[:8]
    logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

    worker = ApiModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_names=[args.model_name],
        limit_worker_concurrency=args.limit_worker_concurrency,
        logger=logger,
        no_register=args.no_register,
        conv_template=None,
        context_len=16384,
        seed=args.seed,
    )

    llm, chunk_process, error_process = get_llm(args.register_api_endpoint_file)
    worker.llm = llm
    worker.chunk_process = chunk_process
    worker.error_process = error_process

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
