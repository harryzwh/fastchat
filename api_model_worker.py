import json
import asyncio
from typing import List, Optional, Dict

from fastchat.conversation import Conversation
from fastchat.serve.base_model_worker import BaseModelWorker
from fastapi import BackgroundTasks


class ApiModelWorker(BaseModelWorker):
    def __init__(
            self,
            controller_addr: str,
            worker_addr: str,
            worker_id: str,
            model_names: List[str],
            limit_worker_concurrency: int,
            conv_template: str = None,
            context_len=2048,
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

        self.context_len = context_len

    def release_worker_semaphore(self):
        self.semaphore.release()

    def acquire_worker_semaphore(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        return self.semaphore.acquire()

    def create_background_tasks(self):
        background_tasks = BackgroundTasks()
        background_tasks.add_task(lambda: self.release_worker_semaphore)
        return background_tasks
