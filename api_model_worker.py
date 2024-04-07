import json
import asyncio
import requests
from typing import List, Optional
from logging import Logger

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.conversation import Conversation
from fastchat import conversation as conv

from fastapi import BackgroundTasks


class ApiModelWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            logger: Logger,
            no_register: bool,
            conv_template: str = None,
            context_len=2048,
            seed: int = None,
            **kwargs,
    ):
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            model_path="",
            model_names=model_names,
            limit_worker_concurrency=limit_worker_concurrency,
        )

        if not no_register:
            self.init_heart_beat()

        url = controller_addr + "/register_worker"
        data = {
            "worker_name": worker_addr,
            "check_heart_beat": not no_register,
            "worker_status": {
                "model_names": model_names,
                "speed": 1,
                "queue_length": self.get_queue_length(),
            },
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

        self.conv = self.make_conv_template(conv_template)
        self.context_len = context_len
        self.llm = None
        self.logger = logger
        self.seed = seed

        logger.info(
            f"Creating {self.model_names} api worker on worker {worker_id} ..."
        )

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

    @staticmethod
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

    @staticmethod
    def chunk_process(chunk):
        raise NotImplemented

    @staticmethod
    def error_process(error):
        raise NotImplemented

    def generate_stream_gate(self, params):
        self.call_ct += 1

        prompt = params["prompt"]
        gen_kwargs = self.get_gen_kwargs(params, seed=self.seed)
        stop = gen_kwargs["stop_sequences"]

        self.logger.info(f"prompt: \n{prompt}")
        self.logger.info(f"gen_kwargs: {gen_kwargs}")

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
                chunked_text, error_code, finish_reason = self.chunk_process(chunk)
                text += chunked_text
                ret = {
                    "text": text,
                    "error_code": error_code,
                    "finish_reason": finish_reason.lower() if finish_reason else None,
                }
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            text, error_code, req_id = self.error_process(e)
            ret = {
                "text": text,
                "error_code": error_code,
                "request_id": req_id,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def release_worker_semaphore(self):
        self.semaphore.release()

    def acquire_worker_semaphore(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        return self.semaphore.acquire()

    def create_background_tasks(self):
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: self.release_worker_semaphore())
        return background_tasks
