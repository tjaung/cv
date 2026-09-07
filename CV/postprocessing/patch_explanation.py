"""Leave-one-patch-out sensitivity for pooled whole-image classifiers."""
import numpy as np


def explain_patches(model, values, records):
    """Positive influence means removing a patch reduces the predicted-class score.

    This is feature omission, not a patch classifier or ground-truth defect mask.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or len(records) != len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite patch feature rows and matching coordinates')
    n = len(values)
    pooled = np.concatenate((values.mean(axis=0), values.std(axis=0)))[None, :]
    prediction = str(model.predict(pooled)[0])
    if n < 2:
        return {'prediction': prediction, 'patches': [], 'reason': 'At least two plate patches are needed for a contribution comparison'}
    total, squared = values.sum(axis=0), (values ** 2).sum(axis=0)
    means = (total-values)/(n-1)
    stds = np.sqrt(np.maximum((squared-values**2)/(n-1)-means**2, 0))
    variants = np.vstack((pooled, np.column_stack((means,stds))))
    classes = model.pipeline.classes_.tolist()
    target = classes.index(prediction)
    if hasattr(model.pipeline, 'decision_function'):
        scores = model.pipeline.decision_function(variants)
        scores = scores[:, target] if scores.ndim == 2 else scores * (1 if target == 1 else -1)
        score_kind = 'decision score'
    else:
        scores = model.pipeline.predict_proba(variants)[:, target]
        score_kind = 'neighbor vote share'
    influence = scores[0] - scores[1:]
    rows = [{**record, 'influence': float(influence[i])} for i, record in enumerate(records)]
    return {'prediction': prediction, 'score_kind': score_kind, 'score': float(scores[0]), 'patches': rows}
