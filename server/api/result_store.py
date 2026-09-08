"""Durable display data/CSVs, separate from opt-in fitted model copies."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

RESULTS_ROOT = Path(__file__).resolve().parents[2] / 'CV/results'
_save_lock = Lock()


def data_root(family):
    return RESULTS_ROOT / 'data' / family


def saved_root(family):
    return RESULTS_ROOT / 'models' / family


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    if path.suffix == '.gz':
        with gzip.open(temp, 'wt') as stream:
            json.dump(value, stream)
    else:
        temp.write_text(json.dumps(value))
    temp.replace(path)


def read_json(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + '.tmp')
    opener = gzip.open(temp, 'wt', newline='', compresslevel=1) if path.suffix == '.gz' else temp.open('w', newline='')
    with opener as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows)
    temp.replace(path)


def image_key(path):
    return hashlib.sha256(path.encode()).hexdigest()


def anomaly_projection_path(model_id, image_path):
    return data_root('anomaly') / model_id / 'predictions' / f'{image_key(image_path)}.json.gz'


def export_anomaly_projection(model_id, image_path, result):
    path = anomaly_projection_path(model_id, image_path)
    write_json(path, {**result, 'model_id': model_id, 'image_id': image_path})
    write_csv(path.with_suffix('').with_suffix('.csv.gz'),
              [{'model_id': model_id, 'image_id': image_path, **row} for row in result['patches']])


def export_anomaly_model(model, report):
    destination = data_root('anomaly') / report['model_id']
    # The UI already subsamples reference scatter points; preserve that displayed
    # resolution, plus every component loading, without duplicating raw matrices.
    detail = {**model.summary(), **report, 'component_weights': model.components.tolist()}
    for key in ('training_points', 'calibration_points'):
        values = detail[key]
        detail[key] = values[::max(1, int(np.ceil(len(values)/1600))) ]
    write_json(destination / 'detail.json.gz', detail)
    write_json(destination / 'report.json', report)
    write_csv(destination / 'metrics.csv.gz', [{**report['config'], **{k:v for k,v in report.items() if k not in ('config','manifest','skipped')}}])
    write_csv(destination / 'components.csv.gz', [{'component': i+1, 'variance_ratio': model.explained_ratio[i], **dict(zip(model.feature_names, weights.tolist()))} for i,weights in enumerate(model.components)])
    for split in ('training', 'calibration'):
        write_csv(destination / f'{split}_plot.csv.gz', [{**{k:v for k,v in p.items() if k != 'scores'}, **{f'PC{i+1}': v for i,v in enumerate(p['scores'])}} for p in detail[f'{split}_points']])


def export_anomaly_evaluation(model_id, evaluation):
    destination = data_root('anomaly') / model_id
    write_json(destination / 'evaluation.json', evaluation)
    write_csv(destination / 'predictions.csv.gz', evaluation['rows'])
    write_csv(destination / 'evaluation_metrics.csv.gz', [{k:v for k,v in evaluation.items() if k not in ('rows','skipped','groups')}])


def export_classifier_summary(directory, summary):
    destination = data_root('classifiers') / directory.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('split.json', 'cv_folds.json', 'features.npz', 'training_features.csv', 'test_features.csv'):
        if (directory / name).exists() and not (destination / name).exists():
            shutil.copy2(directory / name, destination / name)
    metrics, predictions, class_metrics = [], [], []
    for report in summary['models']:
        row = {'model_id': report['id'], **report['config']}
        for split, key in (('training','training_evaluation'),('holdout','evaluation')):
            evaluation = report.get(key)
            if not evaluation:
                continue
            row.update({f'{split}_{m}': evaluation[m] for m in ('accuracy','balanced_accuracy')})
            row[f'{split}_macro_f1'] = evaluation['report']['macro avg']['f1-score']
            samples = summary['train' if split == 'training' else 'test']
            predictions.extend({'model_id': report['id'], 'membership': split, **s, 'prediction': p, 'correct': p == s['label']} for s,p in zip(samples, evaluation['predictions']))
            for label in summary['classes']:
                class_metrics.append({'model_id':report['id'], 'membership':split, 'class':label, **evaluation['report'][label]})
        for metric, score in report['cv'].items():
            row.update({f'cv_{metric}_{stat}': value for stat,value in score.items()})
        metrics.append(row)
    write_csv(destination / 'metrics.csv', metrics)
    write_csv(destination / 'predictions.csv', predictions)
    write_csv(destination / 'class_metrics.csv', class_metrics)
    write_csv(destination / 'split.csv', [{'membership':split, **s} for split,key in [('training','train'),('holdout','test')] for s in summary[key]])
    write_json(destination / 'summary.json', summary)
    write_json(data_root('classifiers') / 'current.json', {'run_id': directory.name})


def export_classifier_plot(directory, model_id, model, summary, train, test):
    z = model.pipeline[:-1].transform(train)
    all_z = np.vstack((z, model.pipeline[:-1].transform(test)))
    xy = np.pad(all_z[:,:2], ((0,0),(0,max(0,2-all_z.shape[1]))))
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = np.maximum((high-low)*.08,.5)
    low, high = low-margin, high+margin
    size = 45
    coords = [low[a]+(np.arange(size)+.5)/size*(high[a]-low[a]) for a in (0,1)]
    xx, yy = np.meshgrid(*coords)
    grid = np.tile(z.mean(axis=0),(size*size,1))
    grid[:,0] = xx.ravel()
    if z.shape[1]>1:
        grid[:,1] = yy.ravel()
    classifier = model.pipeline['classifier']
    predictions = classifier.predict(all_z).tolist()
    neighbors = [[] for _ in all_z]
    if hasattr(classifier, 'kneighbors'):
        distances, indices = classifier.kneighbors(all_z)
        neighbors = [[{'index':int(i),'distance':float(d),**summary['train'][i]} for i,d in zip(row,dist)] for row,dist in zip(indices,distances)]
    samples = [{**s,'point':p.tolist(),'prediction':v} for s,p,v in zip(summary['train']+summary['test'],xy,predictions)]
    snapshot = {'training':samples[:len(train)], 'samples':samples, 'neighbors':neighbors,
                'support_indices':classifier.support_.tolist() if hasattr(classifier,'support_') else [],
                'regions':classifier.predict(grid).reshape(size,size).tolist(), 'bounds':[low.tolist(),high.tolist()],
                'variance':model.pipeline['pca'].explained_variance_ratio_[:2].tolist(), 'dimensions':z.shape[1]}
    destination = data_root('classifiers') / directory.name
    write_json(destination / 'plots' / f'{model_id}.json.gz', snapshot)
    write_csv(destination / 'plots' / f'{model_id}.csv', [{**{k:v for k,v in s.items() if k != 'point'},'PC1':s['point'][0],'PC2':s['point'][1]} for s in samples])


class SaveModelRequest(BaseModel):
    family: Literal['anomaly','classifiers']
    model_id: str
    run_id: str | None = None


def validate_id(value):
    if not value or len(value)!=32 or any(c not in '0123456789abcdef' for c in value):
        raise HTTPException(400,'Invalid result ID')


def save_selected_model(body: SaveModelRequest):
    validate_id(body.model_id)
    with _save_lock:
        if body.family == 'anomaly':
            from .models import ARTIFACT_ROOT
            source = ARTIFACT_ROOT / body.model_id
            destination = saved_root('anomaly') / body.model_id
            if (destination / 'model.npz').exists():
                return {'saved':True}
            if (source / '.pending').exists() or not (source / 'report.json').exists():
                raise HTTPException(409,'Model has not finished saving')
            destination.parent.mkdir(parents=True,exist_ok=True)
            temp = destination.with_name(destination.name+'.tmp')
            try:
                temp.mkdir(exist_ok=True)
                for name in ('model.npz','model.json','report.json'):
                    shutil.copy2(source / name,temp / name)
                temp.replace(destination)
            except Exception:
                shutil.rmtree(temp,ignore_errors=True)
                raise
        else:
            from .classifiers import ROOT
            validate_id(body.run_id)
            source = ROOT / body.run_id / f'{body.model_id}.joblib'
            destination = saved_root('classifiers') / body.run_id / source.name
            if destination.exists():
                return {'saved':True}
            if not source.exists():
                raise HTTPException(404,'Fitted model was deleted; retrain it before saving')
            destination.parent.mkdir(parents=True,exist_ok=True)
            temp = destination.with_suffix('.tmp')
            try:
                shutil.copy2(source,temp)
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
        return {'saved':True}
