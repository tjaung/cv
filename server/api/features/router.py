from fastapi import APIRouter
from .handlers import get_features, get_feature_histograms

router = APIRouter()

router.add_api_route('/features/', get_features, methods=['GET'], tags=['Features'])
router.add_api_route('/features/histograms/', get_feature_histograms, methods=['GET'], tags=['Features'])
