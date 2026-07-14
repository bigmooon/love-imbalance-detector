import pytest
from pydantic import ValidationError

from server.schemas import AnalyzeRequest, ReportPayload, JobStatus


def test_analyze_request_defaults():
    req = AnalyzeRequest(
        upload_id="u1", me="지언",
        start_date="2025-01-01", end_date="2025-06-01",
    )
    assert req.session_gap == 30
    assert req.preset == "기본"
    assert req.api_key is None


def test_analyze_request_rejects_bad_session_gap():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            upload_id="u1", me="지언",
            start_date="2025-01-01", end_date="2025-06-01",
            session_gap=5,  # 최소 10
        )


def test_job_status_literal():
    js = JobStatus(job_id="j1", status="running", step=2, total_steps=7, label="모델 로딩 중")
    assert js.result is None
    with pytest.raises(ValidationError):
        JobStatus(job_id="j1", status="banana", step=0, total_steps=7, label="x")


def test_report_payload_roundtrip(sample_payload_dict):
    payload = ReportPayload.model_validate(sample_payload_dict)
    assert payload.me == "지언"
    assert len(payload.radar.categories) == 7
    assert payload.llm is None
