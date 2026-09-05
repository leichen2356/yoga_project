# Training schemas

This folder sits alongside `model/` at the project root. Open SVG files in a browser to view or print the diagrams. Markdown files include editable Mermaid diagrams and explanatory notes.

| Method | Diagram | Details |
|---|---|---|
| 1. SVM | [SVG](1_svm.svg) | [Markdown](1_svm.md) |
| 2. Random Forest | [SVG](2_random_forest.svg) | [Markdown](2_random_forest.md) |
| 3. Logistic Regression | [SVG](3_logistic_regression.svg) | [Markdown](3_logistic_regression.md) |
| 4. LightGBM | [SVG](4_lightgbm.svg) | [Markdown](4_lightgbm.md) |
| 5. XGBoost | [SVG](5_xgboost.svg) | [Markdown](5_xgboost.md) |
| 6. GCN | [SVG](6_gcn.svg) | [Markdown](6_gcn.md) |

All six describe subtype classification in notebooks 7, 8 and 9 using `subpose_label`. Training has nine subtypes; full test evaluation includes two additional unseen subtypes. GCN uses the shared architecture and backend loader in `code/gcn_model.py`. These schemas document training workflows; they do not train models.

Regenerate the diagrams using `python schema/generate_schemas.py` from the project root. The generator uses Python's standard library and the saved code 7 feature manifest.
