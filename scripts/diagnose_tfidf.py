"""Compare fixed refits and feature selection without changing historical files."""

import argparse
import hashlib
import json
import platform
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from data import model_text
from state import lines, read, sha
from models import fit, score
from measurement import threshold_sweep


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def diagnose(data):
    index = read(ROOT / 'data/split-index.json')
    for split in ('train', 'validation', 'test'):
        if sha(data / (split + '.jsonl')) != index['split_sha256'][split]:
            raise ValueError('original split checksum mismatch: ' + split)
    cfg = read(ROOT / 'models/wj-tfidf/config.json')['config']
    train = lines(data / 'train.jsonl')
    val = lines(data / 'validation.jsonl')
    old = joblib.load(ROOT / 'models/wj-tfidf/model.joblib')
    expected = lines(ROOT / 'evidence/wildjailbreak/wj-tfidf/validation-predictions.jsonl')
    if [r['id'] for r in val] != [r['id'] for r in expected]:
        raise ValueError('validation order mismatch')
    historical = np.array([r['score'] for r in expected])
    texts = [model_text(r, cfg['max_chars'])[0] for r in train]
    counts = CountVectorizer(ngram_range=(1, 2), dtype=np.float64)
    matrix = counts.fit_transform(texts)
    names = counts.get_feature_names_out()
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    cap = cfg['max_features']
    selected = (-totals).argsort()[:cap]
    cutoff = float(totals[selected].min())
    native = fit(train, cfg)
    old_features = set(old['tfidf'].vocabulary_)
    new_features = set(native['tfidf'].vocabulary_)
    lookup = dict(zip(names, totals))
    difference = old_features ^ new_features
    result = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'disabled_cpu_features': os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),
        'enabled_cpu_features': sorted(k for k, v in
            np._core._multiarray_umath.__cpu_features__.items() if v),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'train_rows': len(train), 'validation_rows': len(val),
        'training_text_sha256': digest(texts), 'all_features': len(names),
        'feature_counts_sha256': digest(list(zip(names.tolist(), totals.tolist()))),
        'cutoff_count': cutoff, 'features_above_cutoff': int((totals > cutoff).sum()),
        'features_at_cutoff': int((totals == cutoff).sum()),
        'slots_at_cutoff': cap - int((totals > cutoff).sum()),
        'vocabulary_overlap': len(old_features & new_features),
        'differing_features': len(difference),
        'all_differences_at_cutoff': all(lookup.get(x) == cutoff for x in difference),
        'native_selection_matches_argsort': set(names[selected]) == new_features,
        'fits': {},
    }
    # Saved vocabulary isolates downstream fitting. It is not feature-selection reproduction.
    conditioned = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                vocabulary=old['tfidf'].vocabulary_)),
        ('classifier', LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'],
                                         random_state=cfg['seed'], solver=cfg['solver'])),
    ])
    with threadpool_limits(limits=1):
        conditioned.fit(texts, [r['label'] for r in train])
    for label, model in [('native', native), ('saved_vocabulary_diagnostic', conditioned)]:
        scores = score(model, val, cfg)
        threshold, sweep = threshold_sweep([r['label'] for r in val], scores, 'recall_at_fpr', 500, .05)
        point = next(x for x in sweep if x['threshold'] == threshold)
        result['fits'][label] = {
            'maximum_score_error': float(np.max(np.abs(scores - historical))),
            'threshold': threshold, 'recall': point['recall'], 'fpr': point['fpr'],
            'idf_maximum_error': float(np.max(np.abs(model['tfidf'].idf_ - old['tfidf'].idf_)))
                if label != 'native' or old_features == new_features else None,
        }
    result['historical_match'] = (
        result['vocabulary_overlap'] == cap
        and result['fits']['native']['maximum_score_error'] <= 1e-12
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--require-historical-match', action='store_true')
    a = p.parse_args(argv)
    if a.output.exists():
        p.error('output exists')
    result = diagnose(a.data)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'historical_match': result['historical_match'],
        'maximum_score_error': result['fits']['native']['maximum_score_error'],
    }))
    if a.require_historical_match and not result['historical_match']:
        raise SystemExit('historical feature selection or scores differ')


if __name__ == '__main__':
    main()
