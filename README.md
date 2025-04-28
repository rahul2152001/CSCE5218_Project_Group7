# VQA_Project

A Vision-and-Language Question Answering (VQA) pipeline comparing two approaches:
1. **ResNet-101 + Bi-LSTM** on FastText‐embedded questions  
2. **ViLT** fine-tuned end-to-end on a COCO-style VQA subset


## ✨ Overview

This project demonstrates two VQA models:

- **ResNet_BiLSTM**  
  Extracts image features via a frozen ResNet-101, embeds questions with FastText + Bi-LSTM, and fuses them with an MLP classifier.

- **ViLT**  
  Uses a pre-trained Vision-and-Language Transformer (ViLT) model, fine-tuned with one-hot labels and mixed precision (AMP).

Both pipelines operate on the same subset of COCO VQA data stored in `sample_data/`.

---

## ⚙️ Dependencies

- **Python** 3.8+  
- **PyTorch** & TorchVision  
- **transformers**, **accelerate**  
- **gensim**  
- **scikit-learn**  
- **tqdm**, **pandas**  
- **Pillow**  


 Data Preparation
Place the COCO-style VQA subset inside sample_data/ (adjacent to this README).

Verify that:

sample_data/images/train2014/COCO_train2014_<img_id>.jpg files exist

sample_data/questions/train2014_questions_subset.json

sample_data/annotations/train2014_annotations_subset.json

sample_data/subsets/train_subset_ids.txt

Usage
1. ResNet + Bi-LSTM Pipeline
Open ResNet_BiLSTM_CSCE5218.ipynb in Jupyter or Colab.

Execute cells in order:

Block 1: Imports, path config, seeds

Block 2: Build records list & answer vocabulary

Block 3: FastText embedding setup

Block 4: Extract & cache ResNet-101 image features

Block 5: Dataset & DataLoader setup

Block 6: BiLSTM_VQA model definition

Block 7: Training loop & epoch metrics

2. ViLT Fine-Tuning Pipeline
Open ViLT_CSCE5218.ipynb in Jupyter or Colab.

Execute cells in order:

Block 1: Imports, path config, seeds

Block 2: Build records list & answer vocabulary

Block 3: Load ViLT processor & build label maps

Block 4: ViltVQADataset & DataLoader

Block 5: Load ViltForQuestionAnswering + optimizer

Blocks 6–7: Fine-tuning loops (epochs 1–7, then 8–10)

Block 8: Evaluation (Top-1 & VQA-soft accuracy)

Block 9: Error analysis by question prefix

Utility Script
fix_widgets.py
Removes stale metadata.widgets entries so GitHub can render notebooks

Contact
Maintainer: Rahul Chaudhary

Email: rc2152001@gmail.com
