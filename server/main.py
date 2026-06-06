"""연애 권력 불균형 진단 API. 실행: uvicorn server.main:app --reload"""
import io
import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.kakao_parser import parse_kakao_chat
from server.analysis import AnalysisOptions, TOTAL_STEPS, run_analysis
from server.serialize import build_report_payload
from server.schemas import AnalyzeRequest, JobStatus, UploadSummary
from server.store import JobStore, UploadStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Love Imbalance Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

uploads = UploadStore()
jobs = JobStore()


@app.post("/api/upload", response_model=UploadSummary)
async def upload(file: UploadFile) -> UploadSummary:
    raw = await file.read()
    try:
        df = parse_kakao_chat(io.BytesIO(raw))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("CSV 파싱 실패")
        raise HTTPException(status_code=400, detail=f"CSV를 읽을 수 없습니다: {e}")

    return UploadSummary(
        upload_id=uploads.put(df),
        users=df["User"].unique().tolist(),
        message_count=len(df),
        first_date=df["Date"].min().date(),
        last_date=df["Date"].max().date(),
    )


def _run_job(job_id: str, df, opts: AnalysisOptions):
    try:
        result = run_analysis(
            df, opts,
            progress_cb=lambda i, label: jobs.set_progress(job_id, step=i, label=label),
        )
        jobs.set_done(job_id, build_report_payload(result))
    except ValueError as e:
        jobs.set_error(job_id, str(e))
    except Exception as e:
        logger.exception("분석 실패")
        jobs.set_error(job_id, f"분석 중 오류가 발생했습니다: {e}")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    df = uploads.get(req.upload_id)
    if df is None:
        raise HTTPException(status_code=404, detail="업로드를 찾을 수 없습니다. 다시 업로드해주세요.")
    if req.me not in df["User"].unique():
        raise HTTPException(status_code=400, detail=f"'{req.me}'는 대화 참여자가 아닙니다.")
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="시작일이 종료일보다 늦습니다.")

    opts = AnalysisOptions(
        me=req.me, start_date=req.start_date, end_date=req.end_date,
        session_gap=req.session_gap, preset=req.preset, api_key=req.api_key,
    )
    job_id = jobs.create(total_steps=TOTAL_STEPS)
    background_tasks.add_task(_run_job, job_id, df, opts)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다.")
    return JobStatus(
        job_id=job.job_id, status=job.status, step=job.step,
        total_steps=job.total_steps, label=job.label,
        result=job.result, error=job.error,
    )
