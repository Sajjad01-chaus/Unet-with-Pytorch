# Polygon Coloring using Conditional U-Net
this is the implementation of conditional Unet for polygon coloring. Implemented with  pytorch from scratch

## 📁 Folder Structure
```
Ayna_ML/
│
├── dataset_Ayna/
│   └── dataset/
│       ├── training/
│       │   ├── inputs/         # Polygon shape input images
│       │   ├── outputs/        # Colored output images
│       │   └── data.json       # Mapping of input, output, and color
│       └── validation/
│           ├── inputs/
│           ├── outputs/
│           └── data.json
│
├── training.py                 # Model training script
├── inference.ipynb            # Inference notebook (Colab-ready)
├── Unet_model.py              # Conditional U-Net model definition
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture
A **Conditional U-Net** is used, where the color information is embedded and injected at multiple layers.

## 📊 Training Summary (WandB)
- **Epochs**: 30
- **Train Loss**: 0.0301
- **Validation Loss**: 0.0171
- **Parameters**: 14.7M
- **Model Size**: ~56.4 MB

## 📈 WandB Visuals
The model was logged using Weights & Biases. Check training insights:
👉 [WandB Run Link](https://wandb.ai/sajjadchaush3-thakur-college-of-engineering-and-technology/polygon-coloring-unet/runs/ggla71ow)

![Training Insights](insights.png)

## 🔍 Inference
To run inference on Colab:
1. Upload your test input image (e.g., `triangle.png`)
2. Load the `best_model.pth` from the `wandb/latest-run/files/`
3. Run the notebook `inference.ipynb`

## 🧾 Requirements
Install using:
```bash
pip install -r requirements.txt
```

## 🔗 Submission Links
- ✅ **Colab Inference Notebook**: [https://colab.research.google.com/drive/1g-KmRLzmwcKrfyJD0bzSM_ahN6YQk14v?usp=sharing]
- ✅ **Model File (.pth)**: [https://drive.google.com/file/d/1ni76e-Xh6k7gX0JcB_UEcXeTFISr-AcS/view?usp=sharing]
- ✅ **WandB Public Link**: [https://wandb.ai/sajjadchaush3-thakur-college-of-engineering-and-technology/polygon-coloring-unet?nw=nwusersajjadchaush3]
- ✅ **Insights Report PDF**: [Attached]

---

© 2025 Sajjad Chaush | 
