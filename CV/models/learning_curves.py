import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def learning_curves(pipeline, features, labels, groups, folds, seed=42,
                    fractions=(.2, .4, .6, .8, 1.), progress=None):
    """Refit copies on nested, class-stratified subsets of each CV train fold.

    No held-out test data is accepted. Points require every fold to succeed;
    undersized KNN subsets are reported, not averaged over fewer folds.
    """
    x, y, groups = np.asarray(features), np.asarray(labels), np.asarray(groups)
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError('Features, labels and groups must have matching lengths')
    orders = []
    for fold, (train, validation) in enumerate(folds):
        train, validation = np.asarray(train, dtype=int), np.asarray(validation, dtype=int)
        if set(groups[train]) & set(groups[validation]):
            raise ValueError('CV training and validation content must be disjoint')
        if set(y[train]) != set(y) or set(y[validation]) != set(y):
            raise ValueError('Every fold must contain every class')
        rng = np.random.default_rng(seed + fold)
        orders.append({label: rng.permutation(sorted(set(groups[train][y[train] == label]))) for label in sorted(set(y))})
    points = []
    for step, fraction in enumerate(fractions):
        if not 0 < fraction <= 1:
            raise ValueError('Training fractions must be in (0, 1]')
        scores, sizes, failures = [], [], []
        for fold, (train, validation) in enumerate(folds):
            if progress:
                progress(step * len(folds) + fold, len(fractions) * len(folds))
            selected_groups = {group for order in orders[fold].values() for group in order[:max(1, int(np.ceil(len(order) * fraction))) ]}
            selected = np.asarray([i for i in train if groups[i] in selected_groups], dtype=int)
            sizes.append(len(selected))
            try:
                candidate = clone(pipeline)
                if getattr(candidate['classifier'], 'n_neighbors', 1) > len(selected):
                    raise ValueError('Subset has fewer images than n_neighbors')
                candidate.fit(x[selected], y[selected])
                predictions = [candidate.predict(x[indices]) for indices in (selected, validation)]
                scores.append({f'{split}_{metric}': float(1 - fn(y[indices], prediction))
                               for split, indices, prediction in zip(('training', 'validation'), (selected, validation), predictions)
                               for metric, fn in [('error', accuracy_score), ('balanced_error', balanced_accuracy_score)]})
            except ValueError as error:
                failures.append(f'Fold {fold + 1}: {error}')
        point = {'fraction': fraction, 'images': float(np.mean(sizes)), 'min_images': min(sizes), 'max_images': max(sizes), 'failures': failures}
        if not failures:
            point['scores'] = {key: {'mean': float(np.mean([s[key] for s in scores])),
                                     'std': float(np.std([s[key] for s in scores])), 'folds': [s[key] for s in scores]}
                               for key in scores[0]}
        points.append(point)
    return {'points': points, 'folds': len(folds), 'seed': seed}
