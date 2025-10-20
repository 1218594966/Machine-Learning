"""Entry point for the FastAPI application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core.data_manager import DatasetManager
from .core.model_manager import ModelManager
from .core.preprocess import preprocess_dataset
from .schemas import PredictRequest, SHAPRequest, TrainRequest

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "uploads"
MODEL_DIR = BASE_DIR.parent / "models"

app = FastAPI(title="ML Model Management Platform")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

datasets = DatasetManager(UPLOAD_DIR)
models = ModelManager(MODEL_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel"}:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    target_path = UPLOAD_DIR / file.filename
    with target_path.open("wb") as buffer:
        buffer.write(await file.read())

    dataset_name = datasets.save_dataset(target_path)
    return JSONResponse({"message": "Dataset uploaded successfully", "dataset_name": dataset_name})


@app.get("/api/datasets")
async def list_datasets() -> JSONResponse:
    info = datasets.list_datasets()
    return JSONResponse({"datasets": info})


@app.post("/api/train")
async def train_model(request: TrainRequest) -> JSONResponse:
    try:
        df = datasets.load_dataset(request.dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    preprocess_result = preprocess_dataset(
        df,
        target_column=request.target_column,
        test_size=request.test_size,
        random_state=request.random_state,
    )

    try:
        artifact = models.train_model(
            name=request.model_name,
            preprocess_result=preprocess_result,
            model_type=request.model_type,  # type: ignore[arg-type]
            **(request.model_params or {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"message": "Model trained successfully", "metrics": artifact.metrics})


@app.get("/api/models")
async def list_models() -> JSONResponse:
    return JSONResponse({"models": models.list_models()})


@app.post("/api/predict")
async def predict(request: PredictRequest) -> JSONResponse:
    try:
        transformed = models.transform_inputs(request.model_name, request.features)
        predictions = models.predict(request.model_name, transformed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"predictions": predictions.tolist()})


@app.post("/api/shap")
async def shap_summary(request: SHAPRequest) -> JSONResponse:
    try:
        summary = models.generate_shap_summary(request.model_name, sample_size=request.sample_size)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(summary)
