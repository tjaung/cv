# Saved results

Future training runs automatically export results here. Existing artifacts are
not migrated; retrain to populate these folders.

- `data/anomaly/<model_id>/`: configuration, metrics and prediction CSVs,
  PCA loadings/reference plots, and compressed per-image patch/plot snapshots.
- `data/classifiers/<run_id>/`: metrics, predictions, class metrics, split and
  feature CSVs; shared feature arrays; and fixed training/decision-region plots.
  Each training run remains in its own directory; `current.json` selects the
  latest run displayed by the client.
- `models/`: fitted models explicitly copied by the **Save model** buttons.

Training still writes temporary fitted models to `artifacts/`. After training
and evaluation finish, you can delete those artifacts: exported tables,
predictions and PCA plots remain available. Selectively save any estimators
you want to use for future inference before deleting the artifacts.

Classifier patch explanations and learning curves are computed on request and
cached in `data/`. Outputs already cached remain viewable after deletion;
computing new ones requires the fitted model. Learning curves also export CSV.
Original dataset images must remain available to display the images themselves.

CSV files support external analysis. JSON snapshots preserve nested chart and
display data so the app can show results without loading fitted models.
Generated contents are ignored by Git.
