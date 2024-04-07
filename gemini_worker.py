import os
import uuid
import argparse
import requests
import uvicorn
import asyncio
import json

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from typing import List, Optional
from api_model_worker import ApiModelWorker

from fastchat.utils import build_logger
from fastchat.conversation import Conversation
from fastchat import conversation as conv

worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

app = FastAPI()


class GeminiWorker(ApiModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            no_register: bool,
            conv_template: str = None,
            context_len=16384,
            seed=None,
            **kwargs,
    ):
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            model_names=model_names,
            limit_worker_concurrency=limit_worker_concurrency,
            logger=logger,
            no_register=no_register,
            conv_template=conv_template,
            context_len=context_len,

        )
        self.seed = seed

        from langchain_google_genai import ChatGoogleGenerativeAI

        os.environ["GOOGLE_API_KEY"] = "AIzaSyDGMxYJDddFGXlg_x3_RcxVpr-B3dqM7Eo"

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            convert_system_message_to_human=True,
            # temperature=gen_kwargs.get("temperature"),
            # top_p=gen_kwargs.get("top_p"),
            # top_k=gen_kwargs.get("top_k") if gen_kwargs.get("top_k") > 0 else None,
        )

    def generate_stream_gate(self, params):
        self.call_ct += 1

        prompt = params["prompt"]
        gen_kwargs = self.get_gen_kwargs(params, seed=self.seed)
        stop = gen_kwargs["stop_sequences"]

        logger.info(f"prompt: \n{prompt}")
        logger.info(f"gen_kwargs: {gen_kwargs}")

        if hasattr(self.llm, "temperature") and 'temperature' in gen_kwargs.keys():
            self.llm.temperature = gen_kwargs.get("temperature")
        if hasattr(self.llm, "top_p") and 'top_p' in gen_kwargs.keys():
            self.llm.temperature = gen_kwargs.get("top_p")
        if hasattr(self.llm, "top_k") and 'top_k' in gen_kwargs.keys():
            self.llm.temperature = gen_kwargs.get("top_k") if gen_kwargs.get("top_k") > 0 else None
        if hasattr(self.llm, "penalty_score") and 'repetition_penalty' in gen_kwargs.keys():
            self.llm.temperature = gen_kwargs.get("repetition_penalty")

        text = ""
        try:
            for chunk in self.llm.stream(prompt, stop=stop):
                text += chunk.content
                finish_reason = chunk.response_metadata.get("finish_reason", None)
                ret = {
                    "text": text,
                    "error_code": 0,
                    "finish_reason": finish_reason.lower() if finish_reason else None,
                }
                # print(streamer.hasNext())
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            ret = {
                "text": e,
                # "error_code": e.error_code,
                # "request_id": e.req_id,
            }
            yield json.dumps(ret).encode() + b"\0"


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

    worker = GeminiWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_names=['gemini-api'],
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
    )

    return args, worker


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
