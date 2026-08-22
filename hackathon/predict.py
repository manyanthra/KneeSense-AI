#!/usr/bin/env python3
"""
Run the trained KL-grade classifier on a new knee X-ray image.

Usage:
    python3 predict.py path/to/xray.png
    python3 predict.py path/to/xray.png --model kl_classifier_final.joblib
"""
import sys, argparse
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
import joblib

KL_NAMES = [
    "KL Grade 0 - No Osteoarthritis",
    "KL Grade 1 - Doubtful OA",
    "KL Grade 2 - Minimal OA",
    "KL Grade 3 - Moderate OA",
    "KL Grade 4 - Severe OA"
]

def load_bundle(path):
    return joblib.load(path)

def featurize(image_path, bundle):
    img = Image.open(image_path).convert('L').resize(bundle['img_size'], Image.LANCZOS)
    arr = np.array(ImageOps.equalize(img), dtype=np.float32) / 255.0
    feat = hog(arr, **bundle['hog_params'])
    return feat.reshape(1, -1)

def predict(image_path, bundle):
    X = featurize(image_path, bundle)
    if bundle.get('needs_scaler') and bundle.get('scaler') is not None:
        X = bundle['scaler'].transform(X)
    model = bundle['model']
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    return pred_idx, proba

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--model', default='kl_classifier_final.joblib')
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    pred_idx, proba = predict(args.image, bundle)

    print(f'Predicted KL grade: {pred_idx} ({KL_NAMES[pred_idx]})')
    if proba is not None:
        print('Class probabilities:')
        for name, p in zip(KL_NAMES, proba):
            bar = '#' * int(p * 40)
            print(f'  {name:14s} {p:5.1%}  {bar}')
