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


def get_gen_kwargs(
        params,
        seed: Optional[int] = None,
):
    stop = params.get("stop", None)
    if isinstance(stop, list):
        stop_sequences = stop
    elif isinstance(stop, str):
        stop_sequences = [stop]
    else:
        stop_sequences = []
    gen_kwargs = {
        "do_sample": True,
        "return_full_text": bool(params.get("echo", False)),
        "max_new_tokens": int(params.get("max_new_tokens", 256)),
        "top_p": float(params.get("top_p", 1.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "stop_sequences": stop_sequences,
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "top_k": params.get("top_k", None),
        "seed": seed,
    }
    if gen_kwargs["top_p"] == 1:
        gen_kwargs["top_p"] = 0.9999999
    if gen_kwargs["top_p"] == 0:
        gen_kwargs.pop("top_p")
    if gen_kwargs["temperature"] == 0:
        gen_kwargs.pop("temperature")
        gen_kwargs["do_sample"] = False
    return gen_kwargs


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
            conv_template=conv_template,
            context_len=context_len,
        )

        logger.info(
            f"Creating {self.model_names} api worker on worker {worker_id} ..."
        )

        if not no_register:
            self.init_heart_beat()

        self.conv = self.make_conv_template(conv_template)
        self.context_len = context_len
        self.seed = seed

    # def get_conv_template(self):
    #     return {"conv": self.conv}

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

    def generate_stream_gate(self, params):
        self.call_ct += 1

        prompt = params["prompt"]
        gen_kwargs = get_gen_kwargs(params, seed=self.seed)
        stop = gen_kwargs["stop_sequences"]

        logger.info(f"prompt: \n{prompt}")
        logger.info(f"gen_kwargs: {gen_kwargs}")

        from langchain_google_genai import ChatGoogleGenerativeAI

        os.environ["GOOGLE_API_KEY"] = "AIzaSyDGMxYJDddFGXlg_x3_RcxVpr-B3dqM7Eo"

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            convert_system_message_to_human=True,
            temperature=gen_kwargs.get("temperature"),
            top_p=gen_kwargs.get("top_p"),
            top_k=gen_kwargs.get("top_k") if gen_kwargs.get("top_k") > 0 else None,
        )

        text = ""
        try:
            for chunk in llm.stream(prompt):
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

        # try:
        #     if self.model_path == "":
        #         url = f"{self.api_base}"
        #     else:
        #         url = f"{self.api_base}/{self.model_path}"
        #     client = InferenceClient(url, token=self.token)
        #     res = client.text_generation(
        #         prompt, stream=True, details=True, **gen_kwargs
        #     )
        #
        #     reason = None
        #     text = ""
        #     for chunk in res:
        #         if chunk.token.special:
        #             continue
        #         text += chunk.token.text
        #
        #         s = next((x for x in stop if text.endswith(x)), None)
        #         if s is not None:
        #             text = text[: -len(s)]
        #             reason = "stop"
        #             break
        #         if could_be_stop(text, stop):
        #             continue
        #         if (
        #                 chunk.details is not None
        #                 and chunk.details.finish_reason is not None
        #         ):
        #             reason = chunk.details.finish_reason
        #         if reason not in ["stop", "length"]:
        #             reason = None
        #         ret = {
        #             "text": text,
        #             "error_code": 0,
        #             "finish_reason": reason,
        #         }
        #         yield json.dumps(ret).encode() + b"\0"
        # except Exception as e:
        #     ret = {
        #         "text": f"{SERVER_ERROR_MSG}\n\n({e})",
        #         "error_code": ErrorCode.INTERNAL_ERROR,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


# def release_worker_semaphore(worker):
#     worker.semaphore.release()
#
#
# def acquire_worker_semaphore(worker):
#     if worker.semaphore is None:
#         worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
#     return worker.semaphore.acquire()
#
# def create_background_tasks(worker):
#     background_tasks = BackgroundTasks()
#     background_tasks.add_task(lambda: release_worker_semaphore(worker))
#     return background_tasks


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

    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": worker.worker_addr,
        "check_heart_beat": not args.no_register,
        "worker_status": {
            "model_names": worker.model_names,
            "speed": 1,
            "queue_length": worker.get_queue_length(),
        },
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200

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
