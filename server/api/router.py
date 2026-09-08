"""Register the feature routers consumed by server.main."""
from fastapi import APIRouter

from .comparison.router import router as comparison_router
from .cnn.router import router as cnn_router
from .anomaly_detection.router import router as anomaly_detection_router
from .features.router import router as features_router
from .preprocessing.router import router as preprocessing_router
from .dataset.router import router as dataset_router
from .classifiers.router import router as classifiers_router
from .shared.router import router as shared_router

api_router = APIRouter()

api_router.include_router(comparison_router)
api_router.include_router(cnn_router)
api_router.include_router(anomaly_detection_router)
api_router.include_router(features_router)
api_router.include_router(preprocessing_router)
api_router.include_router(dataset_router)
api_router.include_router(classifiers_router)
api_router.include_router(shared_router)
