"""Finish checks from saved Laya weights; never retrain."""

import argparse
import os
from pathlib import Path
import sys

os.environ['USE_TF'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def finish(data, result):
    import numpy as np
    import torch
    from laya.common import collate_items
    from laya.train import _forward, encode_item, items_from_rows
    from data import model_text
    from early_stopping import EarlyStopping
    from state import dump, lines, read, sha, tree, utc
    from experiment_laya import QUESTIONS, checked_splits, load_eager_checkpoint, operating_point, paired_errors

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    rows = checked_splits(data)['validation']
    saved = lines(result / 'validation-predictions.jsonl')
    if [r['id'] for r in rows] != [r['id'] for r in saved]:
        raise ValueError('prediction order mismatch')
    model, tokenizer, _ = load_eager_checkpoint(result / 'best')
    model.to('cuda').eval()
    items, skipped = items_from_rows(tokenizer, [
        {'state': model_text(r, 12000)[0], 'questions': QUESTIONS,
         'expected': {'harm': 'B' if r['label'] else 'A'}} for r in rows
    ], 512, 192)
    if skipped or len(items) != len(rows):
        raise ValueError('validation rows skipped')
    scores = []
    with torch.inference_mode():
        for start in range(0, len(items), 8):
            chunk = [encode_item(tokenizer, x, 512, 192) for x in items[start:start + 8]]
            logits = _forward(model, collate_items([chunk], tokenizer.pad_token_id), torch.device('cuda'), True, False)
            scores.extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
    scores = np.asarray(scores)
    difference = float(np.max(np.abs(scores - [r['score'] for r in saved])))
    threshold = read(result / 'best/threshold.json')['threshold']
    changed = int(np.sum((scores >= threshold) != [r['prediction'] for r in saved]))
    if difference > 1e-6 or changed:
        raise ValueError('saved reload mismatch')
    selection = read(result / 'best-selection.json')
    original = lines(result / 'selected-fp32-predictions.jsonl')
    stopping = EarlyStopping(patience=2, min_delta=0)
    epochs = read(result / 'epochs.json')
    for epoch in epochs:
        stopping.observe(epoch['epoch'], epoch['validation'])
    report = {
        'status': 'completed', 'date_utc': utc(), 'protocol': read(result / 'protocol.json'),
        'baseline': read(result / 'baseline.json'), 'validation': operating_point(rows, scores),
        'early_stopping': stopping.summary('epoch_limit'),
        'training_seconds': epochs[-1]['elapsed_training_seconds'],
        'serialization': {
            'maximum_score_difference': float(np.max(np.abs(scores - [r['score'] for r in original]))),
            'changed_decisions_at_training_threshold': int(np.sum((scores >= selection['validation']['threshold']) != [r['prediction'] for r in original])),
            'threshold_refitted_on_saved_artifact': True,
        },
        'saved_reload': {'maximum_score_difference': difference, 'changed_decisions': changed},
        'paired_historical_deberta': paired_errors(rows, scores, threshold,
            lines(ROOT / 'evidence/wildjailbreak/wj-deberta-6ep/validation-predictions.jsonl'),
            read(ROOT / 'models/wj-deberta-6ep/threshold.json')['threshold']),
        'length_audit': read(result / 'length-audit.json'), 'test_evaluated': False,
        'checkpoint_sha256': tree(result / 'best'),
        'recovery': {'reason': 'report expected prediction field in historical score-only rows',
                     'script_sha256': sha(Path(__file__)), 'comparison_fix_sha256': sha(ROOT / 'scripts/experiment_laya.py'),
                     'retrained': False, 'unrecorded': ['original peak GPU memory', 'original total seconds']},
    }
    dump(result / 'report.json', report)
    dump(result / 'status.json', {'status': 'completed', 'finished': utc()})
    print('Completed saved checks:', report['validation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    finish(args.data, args.result)
