"""One-class SVM grid experiments using the shared model artifact workflow."""
from itertools import product
from typing import Literal

from pydantic import BaseModel, Field

from CV.models import OneClassSVMDetector
from CV.models.anomaly_detection.inputs import PreprocessingVariant
from . import models


class SVMGridRequest(BaseModel):
    preprocessing: PreprocessingVariant = 'full'
    feature_set: Literal['lab_sobel_hog_frangi'] = 'lab_sobel_hog_frangi'
    patch_size: int = Field(default=64, ge=16, le=256)
    variance_target: float = Field(default=.95, gt=0, lt=1)
    kernels: list[Literal['rbf', 'linear']] = Field(default=['rbf', 'linear'], min_length=1, max_length=2)
    nus: list[float] = Field(default=[.01, .05, .1], min_length=1, max_length=5)
    gammas: list[Literal['scale'] | float] = Field(default=['scale', .01, .1], min_length=1, max_length=5)
    seed: int = Field(default=42, ge=0)


def train_svm_grid(body: SVMGridRequest):
    from fastapi import HTTPException
    import math
    if any(not math.isfinite(nu) or not 0 < nu <= 1 for nu in body.nus):
        raise HTTPException(422, 'nu must be in (0, 1]')
    if any(gamma != 'scale' and (not math.isfinite(gamma) or gamma <= 0) for gamma in body.gammas):
        raise HTTPException(422, 'gamma must be scale or a positive number')
    combinations = [(kernel, nu, gamma) for kernel, nu in product(dict.fromkeys(body.kernels), dict.fromkeys(body.nus))
                    for gamma in (dict.fromkeys(body.gammas) if kernel == 'rbf' else ['scale'])]
    def run(job_id):
        completed, failures, cache = [], [], {}
        for index, (kernel, nu, gamma) in enumerate(combinations):
            with models._lock:
                models._jobs[job_id].update(combination=index + 1, combinations=len(combinations))
            try:
                trained = models._train_pca(models.TrainRequest(patch_size=body.patch_size,
                    variance_target=body.variance_target, seed=body.seed, feature_set=body.feature_set, preprocessing=body.preprocessing), job_id, cache,
                    model_factory=lambda: OneClassSVMDetector(body.variance_target, body.patch_size, nu, kernel, gamma, feature_set=body.feature_set, preprocessing=body.preprocessing))
                models._evaluate_pca(trained['model_id'], job_id, cache)
                completed.append(trained['model_id'])
            except ValueError as error:
                failures.append({'parameters': {'kernel': kernel, 'nu': nu, 'gamma': gamma}, 'error': str(error)})
        return {'model_ids': completed, 'failures': failures}
    return models._submit('svm_grid', run)
