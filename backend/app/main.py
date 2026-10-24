"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rdkit import RDLogger

from app.api.routes import router
from app.core.config import cors_origins
from app.ml.predictor import Predictor

# Invalid user SMILES are returned as 422 responses; keep RDKit from also flooding the server log.
RDLogger.DisableLog("rdApp.*")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.predictor = Predictor()
    yield


app = FastAPI(
    title="Tox21 Explainable Toxicity Screening API",
    description="Research prototype for exploratory predictions on all 12 Tox21 endpoints.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition", "X-Input-Rows", "X-Valid-Rows", "X-Invalid-Rows", "X-Alert-Rows"],
)
app.include_router(router, prefix="/api")