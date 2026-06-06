"""업로드된 DataFrame과 분석 잡의 인메모리 저장소 (단일 프로세스용)."""
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any


class UploadStore:
    """업로드 CSV의 파싱 결과 DataFrame 보관. max_items 초과 시 가장 오래된 것 축출."""

    def __init__(self, max_items: int = 20):
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._max_items = max_items
        self._lock = threading.Lock()

    def put(self, df) -> str:
        upload_id = uuid.uuid4().hex
        with self._lock:
            self._items[upload_id] = df
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return upload_id

    def get(self, upload_id: str):
        with self._lock:
            return self._items.get(upload_id)


@dataclass(frozen=True)
class Job:
    job_id: str
    total_steps: int
    status: str = "pending"
    step: int = 0
    label: str = "대기 중"
    result: Any = None
    error: str | None = None


class JobStore:
    """분석 잡 상태 보관. 갱신은 항상 새 Job 객체로 교체(불변)."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, total_steps: int) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id, total_steps=total_steps)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, **changes)

    def set_progress(self, job_id: str, step: int, label: str):
        self._update(job_id, status="running", step=step, label=label)

    def set_done(self, job_id: str, result):
        self._update(job_id, status="done", result=result, label="완료")

    def set_error(self, job_id: str, error: str):
        self._update(job_id, status="error", error=error, label="오류")
