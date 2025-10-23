"""Entry point for the FastAPI application."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core.data_manager import DatasetManager
from .core.model_manager import ModelManager
from .core.preprocess import (
    FeatureEngineeringConfig,
    preprocess_dataset,
    preview_transformation,
)
from .schemas import (
    DatasetProfile,
    PredictRequest,
    PreprocessPreviewRequest,
    SHAPRequest,
    TrainRequest,
)
from .utils.json_utils import dataframe_to_records, sanitize_for_json

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "uploads"
MODEL_DIR = BASE_DIR.parent / "models"

app = FastAPI(title="ML Model Management Platform")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/guides", StaticFiles(directory=BASE_DIR.parent / "docs"), name="guides")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

datasets = DatasetManager(UPLOAD_DIR)
models = ModelManager(MODEL_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls", ".xlsm"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 CSV 或 Excel 格式的文件")

    temp_name = f"__tmp__{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOAD_DIR / temp_name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with temp_path.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        dataset_name = datasets.save_dataset(temp_path, name=file.filename)
    except ValueError as exc:  # unsupported format or parse error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()

    df = datasets.load_dataset(dataset_name, refresh=True)
    preview = dataframe_to_records(df.head(20))
    return JSONResponse(
        sanitize_for_json(
            {
                "message": "Dataset uploaded successfully",
                "dataset_name": dataset_name,
                "preview": preview,
            }
        )
    )


@app.get("/api/datasets")
async def list_datasets() -> JSONResponse:
    info = datasets.list_datasets()
    return JSONResponse(sanitize_for_json({"datasets": info}))


@app.get("/api/datasets/{name}")
async def get_dataset_profile(name: str) -> JSONResponse:
    try:
        profile = datasets.dataset_profile(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(sanitize_for_json(DatasetProfile(**profile).dict()))


@app.delete("/api/datasets/{name}", status_code=204)
async def delete_dataset(name: str) -> Response:
    try:
        datasets.remove_dataset(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' does not exist")
    return Response(status_code=204)


@app.post("/api/preview")
async def preview_dataset(request: PreprocessPreviewRequest) -> JSONResponse:
    try:
        df = datasets.load_dataset(request.dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    options = FeatureEngineeringConfig.from_options(request.feature_engineering)
    try:
        preview = preview_transformation(
            df,
            feature_columns=request.feature_columns,
            options=options,
            sample_size=request.sample_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(sanitize_for_json(preview))


@app.post("/api/train")
async def train_model(request: TrainRequest) -> JSONResponse:
    try:
        df = datasets.load_dataset(request.dataset_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    options = FeatureEngineeringConfig.from_options(request.feature_engineering)

    try:
        preprocess_result = preprocess_dataset(
            df,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            options=options,
            test_size=request.test_size,
            random_state=request.random_state,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        artifact = models.train_model(
            name=models.generate_model_name(request.dataset_name, request.mode),
            preprocess_result=preprocess_result,
            mode=request.mode,  # type: ignore[arg-type]
            algorithm=request.algorithm,  # type: ignore[arg-type]
            **(request.model_params or {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    overview = models.model_overview(artifact.name)
    return JSONResponse(
        sanitize_for_json(
            {
                "message": "Model trained successfully",
                "metrics": artifact.metrics,
                "model": overview,
            }
        )
    )


@app.get("/api/models")
async def list_models() -> JSONResponse:
    return JSONResponse(sanitize_for_json({"models": models.list_models()}))


@app.get("/api/models/{name}")
async def get_model_details(name: str) -> JSONResponse:
    try:
        overview = models.model_overview(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(sanitize_for_json(overview))


@app.delete("/api/models/{name}", status_code=204)
async def delete_model(name: str) -> Response:
    try:
        models.model_overview(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    models.delete_model(name)
    return Response(status_code=204)


@app.post("/api/predict")
async def predict(request: PredictRequest) -> JSONResponse:
    try:
        transformed = models.transform_inputs(request.model_name, request.features)
        prediction_payload = models.predict_with_proba(request.model_name, transformed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_payload = {"predictions": prediction_payload["predictions"].tolist()}
    probabilities = prediction_payload.get("probabilities")
    if probabilities is not None:
        response_payload["probabilities"] = (
            probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
        )
    return JSONResponse(sanitize_for_json(response_payload))


@app.post("/api/shap")
async def shap_summary(request: SHAPRequest) -> JSONResponse:
    try:
        summary = models.generate_shap_summary(request.model_name, sample_size=request.sample_size)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(sanitize_for_json(summary))


@app.post("/api/reset", status_code=204)
async def reset_state() -> Response:
    datasets.reset_storage()
    models.reset_storage()
    return Response(status_code=204)
