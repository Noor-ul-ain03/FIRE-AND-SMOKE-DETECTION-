# Fire and Smoke Detection from Images (CT-361 CCA)

Classifies a photograph with two independent yes/no answers: **is fire visible?** and **is smoke visible?**
Both can be yes at the same time. Author: (CT-24066), NED University of Engineering and Technology.

## What is in this project

| Stage | What it does |
|---|---|
| Data audit and cleaning | Checks the Kaggle archive, removes 9 unusable files, converts box labels to two yes/no labels |
| Duplicate grouping | Finds near-identical photos (perceptual hash) so they cannot leak between train and test |
| Split | Grouped, stratified train / validation / test split (1,797 / 361 / 360 images) |
| Classical baseline | 74 hand-made colour and texture features, four classical models, grouped 5-fold cross-validation, tuning |
| Neural networks | CNN from scratch, ResNet18 and EfficientNet-B0 (ImageNet weights, fine-tuned) |
| Ensemble and thresholds | Average of ResNet18 and EfficientNet-B0; thresholds tuned on validation data (fire recall of at least 98%) |
| Evaluation and figures | One-time test evaluation, confusion matrices, prediction grids, a small human-check rule |
| Interface | A desktop window or a browser page: upload a photo, see both probabilities and the decision |

## Results on the test set (360 images, used once)

| Model | Mean AUC | Fire recall | Smoke recall |
|---|---|---|---|
| CNN from scratch | 0.803 | 0.997 | 0.583 |
| Classical baseline (Gradient Boosting) | 0.937 | 0.993 | 0.758 |
| EfficientNet-B0 | 0.968 | 0.980 | 0.902 |
| ResNet18 | 0.978 | 0.987 | 0.909 |
| **Ensemble ResNet18 + EfficientNet-B0 (final)** | **0.980** | **0.980** | **0.939** |

The final ensemble missed 6 of 297 fires (0 false alarms) and 8 of 132 smoke images (23 false alarms).
The test set is small, so differences below about 0.01 are not meaningful. See the report for limitations.

## Setup (Windows, Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The dataset is not included. Download **Fire and Smoke Dataset** (`gurneetsingh1928/fire-smoke-dataset-combined`, MIT license) from Kaggle and keep the zip as `archive (1).zip` in the project folder.

## Run the interface (needs the trained models in `models/`)

```powershell
python src/app_tk.py                     # desktop window, nothing extra to install
python src/app_gradio.py                 # browser page, needs: pip install gradio
python src/predict.py "photo.jpg"        # one photo, printed result
python src/predict_folder.py "test images"   # every photo in a folder, saved to outputs/folder_predictions.csv
```

Both interfaces run on your own computer and need no internet once the packages are installed.

## Run the whole pipeline, in order

```powershell
python src/explore_structure.py "archive (1).zip"     # what is inside the zip
python src/audit_yolo_zip.py "archive (1).zip"        # class counts, unreadable images
python src/check_duplicates.py "archive (1).zip"      # near-duplicate report
python src/build_manifest.py "archive (1).zip"        # data/manifest.csv, data/images/
python src/make_splits.py                             # data/splits.csv
python src/preview_clusters.py                        # data/cluster_check.png
python src/dataset.py                                 # checks the data pipeline
python src/features.py                                # data/features.npz (74 features)
python src/classical_baselines.py                     # four classical models, grouped CV
python src/tune_classical.py                          # models/classical_best.joblib
python src/make_colab_zip.py                          # fire_project.zip for Google Colab
```

Neural-network training needs a GPU. On Google Colab (GPU runtime), unzip `fire_project.zip`, then run:

```
python src/train.py --model simple_cnn --epochs 25 --run cnn_scratch
python src/train.py --model resnet18 --epochs 15 --run resnet18
python src/train.py --model efficientnet_b0 --epochs 15 --run efficientnet_b0
```

Copy the resulting `models/*.pt` and `outputs/*` files back to the project, then:

```powershell
python src/final_eval.py          # ensembles, thresholds, test evaluation
python src/make_figures.py        # results figures in outputs/figures/, outputs/test_decisions.csv
python src/make_eda_figures.py    # dataset charts in outputs/eda/
python src/report_facts.py        # collects every number used in the report into outputs/report_facts.txt
```

A fixed random seed (42) is used, so results are reproducible.

## Folders

- `src/` code. `data/` cleaned images and tables. `models/` trained models and `thresholds.json`. `outputs/` results, curves and figures. `test images/` a few look-alike photos used for an informal demonstration.

## Known limitations

- Neither dataset contains photos without fire or smoke, so the model has never seen a normal scene. In an informal check it raised confident false alarms on a lit electric lamp, a sunset and fog.
- One dataset of about 2,500 images; a second download overlapped heavily with it and was excluded.
- Image-level labels come from bounding boxes, so very small boxes can make a whole image positive.
- The system is a decision-support tool and must not replace certified fire alarms or human supervision.
