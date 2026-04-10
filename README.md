# 🎨 NeuralColorizer

**M.Tech Data Science Mini-Project | Rajiv Gandhi Institute of Technology (RIT), Kottayam**

A deep learning application that automatically restores realistic colors to grayscale landscape photographs. Built with a custom CNN architecture featuring a **Self-Attention bottleneck**, the model operates in the CIE LAB color space to predict perceptually accurate colors while preserving the original luminance channel.

---

## ✨ Features

- **Attention-Driven Colorization** — A Self-Attention mechanism in the bottleneck captures long-range spatial context, helping the model distinguish sky, water, foliage, and terrain without "color bleeding."
- **LAB Color Space Pipeline** — The luminance channel (*L*) is preserved as-is; only the *ab* (chrominance) channels are predicted, giving structurally faithful results.
- **Skip Connections** — Encoder feature maps are added back during decoding for sharper spatial reconstruction.
- **Interactive Streamlit UI** — Upload an image, choose inference resolution, and get side-by-side before/after results in a browser.
- **GPU Accelerated** — Automatically uses CUDA if available; falls back to CPU otherwise.

---

## 🧠 Model Architecture

The `Colorizer` network (`model.py`) follows an encoder-decoder design:

```
Input (L channel, 1×H×W)
       │
   [Encoder]
   Conv 1→64   (stride 2)  + BatchNorm + ReLU
   Conv 64→128 (stride 2)  + BatchNorm + ReLU
       │
   [Bottleneck]
   Conv 128→256 (stride 2) + ReLU + SelfAttention(256)
       │
   [Decoder]
   ConvTranspose 256→128   + skip from enc2
   ConvTranspose 128→64    + skip from enc1
   ConvTranspose 64→2      + Tanh  →  predicted ab channels
       │
Output (ab channels, 2×H×W) → reconstructed RGB image
```

The `SelfAttention` module computes query, key, and value projections, applies a softmax attention map, and blends the attended output with the identity via a learnable `gamma` scalar.

---

## 📦 Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/NeuralColorizer.git
cd NeuralColorizer

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `torch` | Model inference |
| `streamlit` | Web UI |
| `Pillow` | Image I/O |
| `scikit-image` | LAB ↔ RGB conversion |
| `numpy` | Array operations |

---

## ▶️ Running the App

Place the trained weights file `landscape_model.pth` in the project root, then launch:

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

1. Use the **Inference Quality** slider in the sidebar to select processing resolution — `128` (faster) or `256` (higher detail).
2. Upload a grayscale (or desaturated) landscape image (JPG/PNG).
3. Click **Generate Color**.
4. The colorized output is displayed alongside the original, and inference time is shown below.

> **Note:** `landscape_model.pth` is excluded from version control (see `.gitignore`). Download or provide the weights separately before running.

---

## 🗂️ Project Structure

```
NeuralColorizer/
├── app.py                  # Streamlit frontend & inference pipeline
├── model.py                # PyTorch model (Colorizer + SelfAttention)
├── landscape_model.pth     # Trained weights (not in repo — add locally)
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## 📊 Training Details

| Setting | Value |
|---|---|
| Dataset | Landscape Colorization Dataset (Kaggle) |
| Epochs | 100+ |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) on *ab* channels |
| Inference resolution | 128×128 or 256×256 (resized back to original for output) |

---

## 🎓 Academic Info

| | |
|---|---|
| **Author** | Swathi P. |
| **Program** | M.Tech Data Science |
| **Institution** | Rajiv Gandhi Institute of Technology (RIT), Kottayam |
| **Year** | 2026 |

---

## 🏋️ Training Your Own Model

The `landscape_model.pth` weights file used by the app **comes from training the notebook on Google Colab**. If you want to retrain the model from scratch — on your own dataset or for more epochs — follow these steps.

---

### Step 1 — Get the Dataset

Download the **Landscape Image Colorization** dataset from Kaggle and upload the color images folder to your Google Drive:

```
MyDrive/
└── NeuralColorizer/
    └── landscape Images/
        └── color/        ← all .jpg/.png training images go here
```

> The dataset used in this project is the [Landscape Image Colorization dataset on Kaggle](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization). Only the `color/` subfolder is needed for training.

---

### Step 2 — Open the Notebook in Google Colab

Upload `Untitled0 (5).ipynb` to Colab (or open it directly from Drive), then set the runtime to **GPU**:

`Runtime → Change runtime type → T4 GPU`

This is essential — training on CPU will be extremely slow.

---

### Step 3 — Run Cell 1 (Mount Drive)

```python
from google.colab import drive
drive.mount('/content/drive')
```

This connects Colab to your Google Drive so that checkpoints are saved there. **Every file saved during training survives a session timeout** because it lives on Drive, not in the temporary Colab VM.

---

### Step 4 — Run Cell 2 (Define the Model)

Just run this cell as-is. It defines the `SelfAttention` and `Colorizer` classes in memory — no changes needed.

---

### Step 5 — Run Cell 3 (Define the Dataset)

Just run this cell as-is. It defines the `LandscapeDataset` class that handles loading images and converting them to LAB format. No changes needed unless your images are in a different folder structure.

---

### Step 6 — Run Cell 4 (Train)

This is the main training cell. Before running, verify the paths at the top match where you put your data:

```python
save_path    = "/content/drive/MyDrive/NeuralColorizer/landscape_model.pth"
dataset_path = '/content/drive/MyDrive/NeuralColorizer/landscape Images/color'
```

Then run the cell. It will:

1. Load the dataset with `batch_size=32`.
2. Check if `landscape_model.pth` already exists on Drive — **if it does, training resumes from that checkpoint** rather than starting from scratch. This means if Colab disconnects mid-training, you just rerun the cell and it picks up where it left off.
3. Train for 50 epochs, printing the loss every 20 steps.
4. **Save the model weights to Drive after every epoch** — so a crash costs at most one epoch of work.

```
Epoch [1/50], Step [0], Loss: 0.0842
Epoch [1/50], Step [20], Loss: 0.0631
...
```

To train for more than 50 epochs total, simply rerun Cell 4 multiple times. Each run loads the previous checkpoint and trains for another 50 epochs on top.

---

### Step 7 — Verify with Cell 6 (Visual Check)

After training, run Cell 6 to visually compare the model's predictions against ground truth on a batch from the dataset:

```
[ Grayscale Input ]  |  [ Model Prediction ]  |  [ Ground Truth ]
```

If the predicted colors look plausible — sky is blue, grass is green — the model has converged well.

---

### Step 8 — Test on Your Own Image with Cell 7

Run Cell 7 to upload any grayscale image from your computer and colorize it:

```python
uploaded = files.upload()   # opens a file picker
```

The cell runs the full inference pipeline (LAB conversion → model → RGB reconstruction) and displays the before/after result inline.

---

### Step 9 — Download the Weights

Once you're happy with the results, download `landscape_model.pth` from your Drive and place it in the root of the cloned repo. The Streamlit app (`app.py`) loads it automatically on startup.

```
NeuralColorizer/
├── app.py
├── model.py
└── landscape_model.pth   ← put it here
```

Then run the app:

```bash
streamlit run app.py
```

---

## 🖼️ Demo

> Add a `demo.png` screenshot of a before/after comparison and uncomment the line below:

![Before and After Demo](demo.png)
