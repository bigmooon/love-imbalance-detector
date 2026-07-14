import pandas as pd

from server.store import UploadStore, JobStore


def test_upload_store_put_get():
    store = UploadStore(max_items=2)
    df = pd.DataFrame({"a": [1]})
    uid = store.put(df)
    assert store.get(uid) is df


def test_upload_store_evicts_oldest():
    store = UploadStore(max_items=2)
    ids = [store.put(pd.DataFrame({"a": [i]})) for i in range(3)]
    assert store.get(ids[0]) is None  # 가장 오래된 것 축출
    assert store.get(ids[2]) is not None


def test_upload_store_missing_returns_none():
    assert UploadStore().get("nope") is None


def test_job_lifecycle():
    store = JobStore()
    job_id = store.create(total_steps=7)
    job = store.get(job_id)
    assert job.status == "pending"

    store.set_progress(job_id, step=2, label="감정 분류 중")
    job = store.get(job_id)
    assert job.status == "running"
    assert job.step == 2
    assert job.label == "감정 분류 중"

    store.set_done(job_id, result={"ok": True})
    job = store.get(job_id)
    assert job.status == "done"
    assert job.result == {"ok": True}


def test_job_error():
    store = JobStore()
    job_id = store.create(total_steps=7)
    store.set_error(job_id, "boom")
    job = store.get(job_id)
    assert job.status == "error"
    assert job.error == "boom"


def test_job_missing_returns_none():
    assert JobStore().get("nope") is None
