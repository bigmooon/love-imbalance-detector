import io

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.main import app

CSV = "Date,User,Message\n2025-01-06 10:00:00,지언,안녕 뭐해?\n2025-01-06 10:01:00,민수,일하지\n"


@pytest.fixture
def client(sample_payload_dict, monkeypatch):
    # run_analysis와 직렬화를 가짜로 대체 (모델 로딩 회피)
    from server.schemas import ReportPayload

    def fake_run(df, opts, progress_cb=lambda i, l: None, **kw):
        progress_cb(0, "데이터 준비 중")
        return {"fake": True}

    monkeypatch.setattr(main, "run_analysis", fake_run)
    monkeypatch.setattr(
        main, "build_report_payload",
        lambda result: ReportPayload.model_validate(sample_payload_dict),
    )
    return TestClient(app)


def _upload(client):
    return client.post(
        "/api/upload",
        files={"file": ("chat.csv", io.BytesIO(CSV.encode("utf-8")), "text/csv")},
    )


def test_upload_returns_summary(client):
    res = _upload(client)
    assert res.status_code == 200
    body = res.json()
    assert set(body["users"]) == {"지언", "민수"}
    assert body["message_count"] == 2
    assert body["first_date"] == "2025-01-06"


def test_upload_invalid_csv_returns_400(client):
    res = client.post(
        "/api/upload",
        files={"file": ("bad.csv", io.BytesIO(b"Date,User,Message\n"), "text/csv")},
    )
    assert res.status_code == 400


def test_analyze_then_poll_job(client):
    upload_id = _upload(client).json()["upload_id"]
    res = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # TestClient는 BackgroundTask를 응답 후 동기 실행하므로 바로 done
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["result"]["me"] == "지언"


def test_analyze_unknown_upload_returns_404(client):
    res = client.post("/api/analyze", json={
        "upload_id": "nope", "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 404


def test_analyze_unknown_user_returns_400(client):
    upload_id = _upload(client).json()["upload_id"]
    res = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "없는사람",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 400


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_analyze_reversed_dates_returns_400(client):
    upload_id = _upload(client).json()["upload_id"]
    res = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "지언",
        "start_date": "2025-12-31", "end_date": "2025-01-01",
    })
    assert res.status_code == 400


def test_validation_error_does_not_echo_api_key(client):
    # me 누락 → 422. 응답 본문에 api_key 값이 반향되면 안 됨
    res = client.post("/api/analyze", json={
        "upload_id": "u1",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "api_key": "sk-secret-test-key",
    })
    assert res.status_code == 422
    assert "sk-secret-test-key" not in res.text


def test_analysis_error_sets_job_error(client, monkeypatch):
    def boom(df, opts, progress_cb=lambda i, l: None, **kw):
        raise ValueError("기간에 메시지가 없습니다")

    monkeypatch.setattr(main, "run_analysis", boom)
    upload_id = _upload(client).json()["upload_id"]
    job_id = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    }).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "error"
    assert "기간" in job["error"]
