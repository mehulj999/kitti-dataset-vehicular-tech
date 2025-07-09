# Object Detection on KITTI Dataset using Faster R-CNN and SSDLite

This project demonstrates training and evaluating two object detection models — **Faster R-CNN** and **SSDLite** — on a reduced version of the KITTI dataset. The pipeline includes data preprocessing, model training (with early stopping), evaluation using multiple metrics, and saving of logs, graphs, and model weights.

---

## 📂 KITTI Dataset

The [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) is a widely used dataset for autonomous driving and computer vision tasks such as object detection, tracking, and scene understanding.

We use the **Object Detection** subset of the KITTI dataset. Due to its size, only 1/20th of the dataset is used in this project for faster training and testing.

### 📥 Download Links

- KITTI Raw Data: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- YOLO-formatted labels (via Kaggle or converted manually)

### 🗂 Folder Structure

The following folder structure is expected:
```
kaggle/input/
├── kitti-dataset/
│   └── data_object_image_2/
│       └── training/
│           └── image_2/             # KITTI training images
├── kitti-dataset-yolo-format/
│   ├── labels/                      # YOLO-formatted label text files
│   └── classes.json                 # Mapping of class IDs to class names
```

Make sure to adjust the paths in the code if your folders are organized differently.

---

## 🧠 Models Used

### 1. **Faster R-CNN**
Faster R-CNN is a two-stage object detection model that first proposes regions of interest and then classifies objects in those regions. It is known for its **high accuracy** and robustness in complex scenes.

### 2. **SSDLite**
SSDLite is a lightweight version of the Single Shot MultiBox Detector (SSD) that uses **MobileNetV3** as a backbone. It is optimized for **real-time inference on edge devices** with limited resources.

These models were chosen to compare **accuracy vs. efficiency** in object detection tasks.

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

1. **Install all dependencies** using:
   ```bash
   pip install -r requirements.txt
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
- 

---

## 📚 References

- **Faster R-CNN:**  
  Ren, S., He, K., Girshick, R., & Sun, J. (2015).  
  *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*.  
  [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

- **SSDLite / MobileNetV3:**  
  Howard, A., Sandler, M., Chu, G., et al. (2019).  
  *Searching for MobileNetV3*.  
  [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)


---

## 📄 License

This project is released under the [MIT License](LICENSE).
