from fastapi import APIRouter
from .handlers import list_classifiers, train_classifiers, inspect_classifier, classify_review_image, get_classifier_training_metrics, get_classifier_curves, train_classifier_curves, get_classifier_patch_explanation

router = APIRouter()

router.add_api_route('/classifiers/', list_classifiers, methods=['GET'], tags=['Classifiers'])
router.add_api_route('/classifiers/train/', train_classifiers, methods=['POST'], tags=['Classifiers'])
router.add_api_route('/classifiers/inspect/', inspect_classifier, methods=['GET'], tags=['Classifiers'])
router.add_api_route('/classifiers/review/', classify_review_image, methods=['GET'], tags=['Classifiers'])
router.add_api_route('/classifiers/training_metrics/', get_classifier_training_metrics, methods=['GET'], tags=['Classifiers'])
router.add_api_route('/classifiers/learning_curves/', get_classifier_curves, methods=['GET'], tags=['Classifiers'])
router.add_api_route('/classifiers/learning_curves/', train_classifier_curves, methods=['POST'], tags=['Classifiers'])
router.add_api_route('/classifiers/patch_explanation/', get_classifier_patch_explanation, methods=['GET'], tags=['Classifiers'])
