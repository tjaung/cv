from fastapi import APIRouter
from .handlers import get_comparison_models, load_comparison_models, run_comparison, get_comparison_run, get_comparison_image

router = APIRouter()

router.add_api_route('/comparison/models/', get_comparison_models, methods=['GET'], tags=['Comparison'])
router.add_api_route('/comparison/load/', load_comparison_models, methods=['POST'], tags=['Comparison'])
router.add_api_route('/comparison/run/', run_comparison, methods=['POST'], tags=['Comparison'])
router.add_api_route('/comparison/results/{run_id}', get_comparison_run, methods=['GET'], tags=['Comparison'])
router.add_api_route('/comparison/images/{run_id}/{image_index}', get_comparison_image, methods=['GET'], tags=['Comparison'])
