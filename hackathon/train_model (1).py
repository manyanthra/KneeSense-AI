"""
Knee OA KL-Grade Classifier — training pipeline.

Data: Kaggle "Knee Osteoarthritis Dataset with Severity Grading" (OAI-derived),
expects the standard train/ val/ test/ folder-of-folders layout with class
subfolders '0'-'4' (KL grades).

Approach: HOG (Histogram of Oriented Gradients) features + classical sklearn
classifiers, benchmarked against each other and evaluated on the dataset's own
held-out test/ split.

Why not a CNN: on hardware without a GPU (e.g. a 1-vCPU sandbox with no route
to download.pytorch.org), the plain PyPI 'torch' wheel pulls multi-GB CUDA
dependencies for no benefit on CPU-only training. This pipeline trains in
under 5 minutes on a single core with no such dependency. Swap in a real CNN
(architecture sketch in README) if you have GPU access, e.g. via Colab.

Usage:
    python3 train_model.py --data_dir /path/to/dataset
"""
import os, time, json, argparse
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_SIZE = (96, 96)
CLASSES = ['0', '1', '2', '3', '4']
KL_NAMES = ['0 - None', '1 - Doubtful', '2 - Minimal', '3 - Moderate', '4 - Severe']
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(12, 12),
                   cells_per_block=(2, 2), block_norm='L2-Hys')


def featurize_split(data_dir, split):
    X, y, paths = [], [], []
    for label in CLASSES:
        folder = os.path.join(data_dir, split, label)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            img = Image.open(fpath).convert('L').resize(IMG_SIZE, Image.LANCZOS)
            arr = np.array(ImageOps.equalize(img), dtype=np.float32) / 255.0
            feat = hog(arr, **HOG_PARAMS)
            X.append(feat); y.append(int(label)); paths.append(fpath)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), paths


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print('=== Featurizing splits ===')
    t0 = time.time()
    X_train, y_train, _ = featurize_split(data_dir, 'train')
    X_val, y_val, _ = featurize_split(data_dir, 'val')
    X_test, y_test, test_paths = featurize_split(data_dir, 'test')
    print(f'train={X_train.shape} val={X_val.shape} test={X_test.shape} ({time.time()-t0:.1f}s)')

    scaler = StandardScaler().fit(X_train)
    sw_train = compute_sample_weight('balanced', y_train)

    print('\n=== Training candidates ===')
    models = {}

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = dict(model=rf, needs_scaler=False)
    print(f'random_forest: {time.time()-t0:.1f}s')

    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=150, random_state=42)
    hgb.fit(X_train, y_train, sample_weight=sw_train)
    models['hist_gradient_boosting'] = dict(model=hgb, needs_scaler=False)
    print(f'hist_gradient_boosting: {time.time()-t0:.1f}s')

    t0 = time.time()
    sgd = SGDClassifier(loss='log_loss', class_weight='balanced',
                         max_iter=1000, tol=1e-3, random_state=42)
    sgd.fit(scaler.transform(X_train), y_train)
    models['sgd_linear'] = dict(model=sgd, needs_scaler=True)
    print(f'sgd_linear: {time.time()-t0:.1f}s')

    print('\n=== Validation comparison (macro-F1, classes are imbalanced) ===')
    val_scores = {}
    for name, m in models.items():
        Xv = scaler.transform(X_val) if m['needs_scaler'] else X_val
        pred = m['model'].predict(Xv)
        acc, f1 = accuracy_score(y_val, pred), f1_score(y_val, pred, average='macro')
        val_scores[name] = f1
        print(f'{name:24s} acc={acc:.3f}  macro_f1={f1:.3f}')

    best_name = max(val_scores, key=val_scores.get)
    best = models[best_name]
    print(f'\nBest on validation: {best_name}')

    print(f'\n=== Final TEST evaluation ({best_name}) ===')
    Xt = scaler.transform(X_test) if best['needs_scaler'] else X_test
    test_pred = best['model'].predict(Xt)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    report = classification_report(y_test, test_pred, target_names=KL_NAMES, digits=3)
    cm = confusion_matrix(y_test, test_pred)
    print(f'accuracy={test_acc:.3f}  macro_f1={test_f1:.3f}')
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f'KL{i}' for i in range(5)])
    ax.set_yticklabels([f'KL{i}' for i in range(5)])
    ax.set_xlabel('Predicted grade'); ax.set_ylabel('True grade')
    ax.set_title(f'Test confusion matrix — {best_name}\nacc={test_acc:.2f}, macro-F1={test_f1:.2f}')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)

    joblib.dump({
        'model_name': best_name,
        'model': best['model'],
        'scaler': scaler if best['needs_scaler'] else None,
        'needs_scaler': best['needs_scaler'],
        'img_size': IMG_SIZE,
        'classes': CLASSES,
        'hog_params': HOG_PARAMS,
    }, os.path.join(out_dir, 'kl_classifier_final.joblib'))

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'validation_macro_f1_by_model': val_scores,
            'best_model': best_name,
            'test_accuracy': float(test_acc),
            'test_macro_f1': float(test_f1),
            'test_confusion_matrix': cm.tolist(),
            'test_classification_report': report,
        }, f, indent=2)

    print(f'\nSaved model + metrics + confusion_matrix.png to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='.')
    ap.add_argument('--out_dir', default='./artifacts')
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)
