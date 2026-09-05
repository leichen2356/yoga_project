"""Regenerate the six training diagrams as SVG and Mermaid Markdown."""
from pathlib import Path
import html
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
manifest = json.loads((ROOT / 'model/code7_subpose_classification/model_manifest.json').read_text())
feature_count = len(manifest['feature_columns'])
classes = ', '.join(manifest['classes'])

specs = [
    ('1_svm', 'SVM (RBF kernel)', 'Implemented — notebook 7',
     ['StandardScaler: fit on training data only', 'Apply the fitted scaler to validation and test'],
     ['SVC: C=3.0, kernel=rbf, gamma=scale', 'class_weight=balanced, probability=True, seed=42'],
     ['Save scaler + classifier as rbf_svm.joblib', 'Output: model/code7_subpose_classification/']),
    ('2_random_forest', 'Random Forest', 'Implemented — notebook 7',
     ['Use keypoint and angle features directly', 'No feature scaling'],
     ['500 trees; max_features=sqrt; min_samples_leaf=1', 'class_weight=balanced, n_jobs=-1, seed=42'],
     ['Save classifier as random_forest.joblib', 'Output: model/code7_subpose_classification/']),
    ('3_logistic_regression', 'Logistic Regression', 'Implemented — notebook 7',
     ['StandardScaler: fit on training data only', 'Apply the fitted scaler to validation and test'],
     ['LogisticRegression: C=1.0, max_iter=3000', 'class_weight=balanced, seed=42'],
     ['Save scaler + classifier as logistic_regression.joblib', 'Output: model/code7_subpose_classification/']),
    ('4_lightgbm', 'LightGBM', 'Implemented — notebook 8',
     ['Use features directly; no scaling', 'Encode subpose_label as class IDs 0–8'],
     ['500 trees; 15 leaves; learning_rate=0.05', 'subsample=0.9, subsample_freq=1, colsample_bytree=0.9',
      'objective=multiclass, n_jobs=4, seed=42'],
     ['Save native models/lightgbm.txt + label_mapping.json', 'Output: model/code8_boosting_subpose_classification/']),
    ('5_xgboost', 'XGBoost', 'Implemented — notebook 8',
     ['Use features directly; no scaling', 'Encode subpose_label as class IDs 0–8'],
     ['500 trees; max_depth=4; learning_rate=0.05', 'subsample=0.9, colsample_bytree=0.9, tree_method=hist',
      'objective=multi:softprob, eval_metric=mlogloss, seed=42'],
     ['Save native models/xgboost.json + label_mapping.json', 'Output: model/code8_boosting_subpose_classification/']),
    ('6_gcn', 'Graph Convolutional Network (GCN)', 'Implemented — notebook 9',
     ['23 keypoints; XYZ per node; training-only feature scaling', 'Body edges + custom nose-to-shoulder links + self-loops',
      'Keep 7 angle features as a separate graph-level input'],
     ['GCN(3→64) + ReLU → GCN(64→64) + ReLU; dropout=0.1',
      'Mean pool nodes → concatenate 7 angles → Dense(71→9)',
      'Train logits with cross-entropy; softmax for inference'],
     ['models/gcn.pt + gcn_model.py + graph/label/scaling metadata', 'Output: model/code9_gcn_subpose_classification/']),
]

for stem, title, status, preprocessing, estimator, saving in specs:
    blocks = [
        ('1. Prepared data', ['Existing train / validation / test CSVs; no new split',
                              f'{feature_count} features: 23 keypoints × XYZ + 7 joint angles',
                              'Target: subpose_label (9 training subtypes); metadata excluded']),
        ('2. Model input', preprocessing),
        ('3. Train on training split', estimator),
        ('4. Validation and selection',
         ['Rank models by validation macro-F1; accuracy breaks ties',
          'Do not fit on validation or select using test scores'] if stem != '6_gcn' else
         ['Choose checkpoint by validation macro-F1, then accuracy',
          'Adam lr=0.001; max 600 epochs; patience=100; batch=64']),
        ('5. Held-out evaluation', ['Predict labels and class probabilities on validation and test',
                                    'Accuracy, macro/weighted precision, recall and F1',
                                    'Per-subtype reports; 11 test labels include 2 unseen subtypes']),
        ('6. Save for backend inference', saving + ['Save feature order and class mapping in model metadata']),
    ]
    mermaid = ['flowchart TD']
    for i, (heading, lines) in enumerate(blocks):
        label = '<br/>'.join([heading, *lines]).replace('"', "'")
        mermaid.append(f'    N{i}["{label}"]')
        if i:
            mermaid.append(f'    N{i-1} --> N{i}')
    note = ('Matches the current notebook configuration. The validation winner is selected among '
            'the models compared by the notebook; test metrics are descriptive. Model files are '
            'inside the output directory’s models/ folder. Best-model metadata identifies the selected model.')
    (HERE / f'{stem}.md').write_text(
        f'# {title} training schema\n\n{status}\n\n![Training schema]({stem}.svg)\n\n'
        + '```mermaid\n' + '\n'.join(mermaid) + '\n```\n\n'
        + f'Classes: {classes}.\n\n{note}\n', encoding='utf-8')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1170" viewBox="0 0 1000 1170">',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#64748b"/></marker></defs>',
           '<rect width="1000" height="1170" fill="#f8fafc"/>',
           f'<text x="50" y="52" font-family="Arial" font-size="29" font-weight="bold" fill="#0f172a">{html.escape(title)} — training schema</text>',
           f'<text x="50" y="83" font-family="Arial" font-size="18" fill="#475569">{html.escape(status)}</text>']
    for i, (heading, lines) in enumerate(blocks):
        y = 110 + 164 * i
        if i:
            svg.append(f'<path d="M500,{y-23} L500,{y-6}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
        svg.append(f'<rect x="50" y="{y}" width="900" height="140" rx="12" fill="white" stroke="#cbd5e1"/>')
        svg.append(f'<text x="75" y="{y+29}" font-family="Arial" font-size="20" font-weight="bold" fill="#0f766e">{html.escape(heading)}</text>')
        for j, line in enumerate(lines):
            svg.append(f'<text x="75" y="{y+57+j*25}" font-family="Arial" font-size="18" fill="#334155">{html.escape(line)}</text>')
    for row in range(3):
        text = ', '.join(manifest['classes'][row * 3:row * 3 + 3])
        svg.append(f'<text x="50" y="{1107 + 20 * row}" font-family="Arial" font-size="15" fill="#475569">{html.escape(text)}</text>')
    svg.append('</svg>')
    (HERE / f'{stem}.svg').write_text('\n'.join(svg), encoding='utf-8')

print('Updated all six subtype diagrams and Markdown schemas.')
