# SVM (RBF kernel) training schema

Implemented — notebook 7

![Training schema](1_svm.svg)

```mermaid
flowchart TD
    N0["1. Prepared data<br/>Existing train / validation / test CSVs; no new split<br/>76 features: 23 keypoints × XYZ + 7 joint angles<br/>Target: subpose_label (9 training subtypes); metadata excluded"]
    N1["2. Model input<br/>StandardScaler: fit on training data only<br/>Apply the fitted scaler to validation and test"]
    N0 --> N1
    N2["3. Train on training split<br/>SVC: C=3.0, kernel=rbf, gamma=scale<br/>class_weight=balanced, probability=True, seed=42"]
    N1 --> N2
    N3["4. Validation and selection<br/>Rank models by validation macro-F1; accuracy breaks ties<br/>Do not fit on validation or select using test scores"]
    N2 --> N3
    N4["5. Held-out evaluation<br/>Predict labels and class probabilities on validation and test<br/>Accuracy, macro/weighted precision, recall and F1<br/>Per-subtype reports; 11 test labels include 2 unseen subtypes"]
    N3 --> N4
    N5["6. Save for backend inference<br/>Save scaler + classifier as rbf_svm.joblib<br/>Output: model/code7_subpose_classification/<br/>Save feature order and class mapping in model metadata"]
    N4 --> N5
```

Classes: downdog_subpose_1, goddess_subpose_2, plank_subpose_1, plank_subpose_2, plank_subpose_4, tree_left_subpose_2, tree_right_subpose_2, warrior2_left_subpose_1, warrior2_right_subpose_1.

Matches the current notebook configuration. The validation winner is selected among the models compared by the notebook; test metrics are descriptive. Model files are inside the output directory’s models/ folder. Best-model metadata identifies the selected model.
