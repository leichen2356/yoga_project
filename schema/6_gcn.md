# Graph Convolutional Network (GCN) training schema

Implemented — notebook 9

![Training schema](6_gcn.svg)

```mermaid
flowchart TD
    N0["1. Prepared data<br/>Existing train / validation / test CSVs; no new split<br/>76 features: 23 keypoints × XYZ + 7 joint angles<br/>Target: subpose_label (9 training subtypes); metadata excluded"]
    N1["2. Model input<br/>23 keypoints; XYZ per node; training-only feature scaling<br/>Body edges + custom nose-to-shoulder links + self-loops<br/>Keep 7 angle features as a separate graph-level input"]
    N0 --> N1
    N2["3. Train on training split<br/>GCN(3→64) + ReLU → GCN(64→64) + ReLU; dropout=0.1<br/>Mean pool nodes → concatenate 7 angles → Dense(71→9)<br/>Train logits with cross-entropy; softmax for inference"]
    N1 --> N2
    N3["4. Validation and selection<br/>Choose checkpoint by validation macro-F1, then accuracy<br/>Adam lr=0.001; max 600 epochs; patience=100; batch=64"]
    N2 --> N3
    N4["5. Held-out evaluation<br/>Predict labels and class probabilities on validation and test<br/>Accuracy, macro/weighted precision, recall and F1<br/>Per-subtype reports; 11 test labels include 2 unseen subtypes"]
    N3 --> N4
    N5["6. Save for backend inference<br/>models/gcn.pt + gcn_model.py + graph/label/scaling metadata<br/>Output: model/code9_gcn_subpose_classification/<br/>Save feature order and class mapping in model metadata"]
    N4 --> N5
```

Classes: downdog_subpose_1, goddess_subpose_2, plank_subpose_1, plank_subpose_2, plank_subpose_4, tree_left_subpose_2, tree_right_subpose_2, warrior2_left_subpose_1, warrior2_right_subpose_1.

Matches the current notebook configuration. The validation winner is selected among the models compared by the notebook; test metrics are descriptive. Model files are inside the output directory’s models/ folder. Best-model metadata identifies the selected model.
