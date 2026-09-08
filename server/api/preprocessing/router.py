from fastapi import APIRouter, Response
from .handlers import get_preprocessing_steps, run_preprocessing_pipeline

router = APIRouter()

router.add_api_route('/preprocessing/steps/', get_preprocessing_steps, methods=['GET'], tags=['Preprocessing'])
router.add_api_route(
    '/preprocessing/pipeline/', run_preprocessing_pipeline, methods=['GET'],
    tags=['Preprocessing'], response_class=Response,
    responses={200: {'content': {'image/png': {}}}},
)
