# Object Detection on KITTI Dataset using Faster R-CNN and SSDLite

This project demonstrates training and evaluating two object detection models — **Faster R-CNN** and **SSDLite** — on a reduced version of the KITTI dataset. The pipeline includes data preprocessing, model training (with early stopping), evaluation using multiple metrics, and saving of logs, graphs, and model weights.

---

## 📂 Dataset

We use the **KITTI Object Detection** dataset with labels in **YOLO format**. Due to the large size of the dataset, only a subset (1/20th) is used to reduce training time for experimentation.

- Image Path: `kaggle/input/kitti-dataset/data_object_image_2/training/image_2/`
- Label Path: `kaggle/input/kitti-dataset-yolo-format/labels/`
- Class Mapping: `kaggle/input/kitti-dataset-yolo-format/classes.json`

> ⚠️ Ensure these paths are valid or adjust them accordingly in the code.

---

## 🧠 Models

Two object detection models from PyTorch’s `torchvision` are used:
- ✅ `Faster R-CNN` with ResNet-50 FPN backbone
- ✅ `SSDLite` with MobileNetV3 Large backbone

---

## 🚀 Features

- ✅ Train each model separately with custom hyperparameters
- ✅ Skip training if pre-trained weights are already saved
- ✅ Evaluate with:
  - Precision, Recall, F1-score (per class)
  - Confusion Matrix
  - ROC AUC Curves
- ✅ Save outputs:
  - `log.txt` — Training loss per epoch
  - `metrics.json` — Evaluation scores
  - `confusion_matrix.png`, `auc_roc.png`, `metrics_plot.png` — Evaluation visualizations
  - `loss_plot.png` — Training loss graph
  - `class_distribution.png` — Distribution of labels in test set
- ✅ All model outputs are saved under `outputs/<model_name>/`

---

## 📊 Output Structure

```
outputs/
├── fasterrcnn/
│   ├── weights.pth
│   ├── log.txt
│   ├── metrics.json
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── auc_roc.png
│   └── metrics_plot.png
└── ssdlite/
    └── (same structure)
```

---

## 🧪 How to Run

1. **Install requirements**:
   ```bash
   pip install torch torchvision matplotlib scikit-learn tqdm
   ```

2. **Ensure dataset folders and `classes.json` are placed as expected.**

3. **Run the script**:
   ```bash
   python new_one.py
   ```

---

## 🛠 Customization

You can tweak the following in the script:
- `EPOCHS`, `BATCH_SIZE`, `PATIENCE`
- Dataset size (`mini_pairs = pairs[:len(pairs)//20]`)
- Model selection via `run_training("fasterrcnn")` or `run_training("ssdlite")`

---

## 📈 Sample Evaluation Plots

![Confusion Matrix](outputs/fasterrcnn/confusion_matrix.png)
![Metrics Plot](outputs/fasterrcnn/metrics_plot.png)
![ROC AUC](outputs/fasterrcnn/auc_roc.png)

---

## 📌 Notes

- GPU is automatically used if available (`torch.cuda.is_available()`).
- Early stopping is used to prevent overfitting.
- The code is modular and easy to extend to other models or datasets.

---

## 🏫 University Submission

This is a group project submitted for the subject **Vehicular Technology** at the **University of Europe for Applied Sciences**, Potsdam.

### 👥 Group Members:
- Mehul Dinesh Jain
- Bora Ozdamar
- Berke
- Ghazale
- Daniel Alexander
- Maryam

---

## 📄 License

This project is released under the [MIT License](LICENSE).
