from fastapi import APIRouter
from .result_store import save_selected_model

router = APIRouter()

router.add_api_route('/results/save_model/', save_selected_model, methods=['POST'], tags=['Results'])
