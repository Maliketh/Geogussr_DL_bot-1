# 🌍 GeoGuessr AI – Predicting Countries from Street View

This project trains a neural network to predict **which country a Google Street View image comes from**, inspired by the game *GeoGuessr*.

The model uses **CLIP’s ResNet50 image encoder** as a backbone, fine-tuned with a custom **geo-aware loss function** that accounts for real-world distances between countries. Unlike standard classification, where all mistakes are punished equally, this model is more forgiving if the prediction is geographically close to the ground truth.

---

## 🔎 Motivation

Traditional classifiers treat all misclassifications the same. For geolocation, this doesn’t make sense:
- Guessing **Serbia** instead of **Croatia** is a small mistake.
- Guessing **Brazil** instead of **Croatia** is a huge one.

This project implements a **distance-weighted loss function** that encodes geographic relationships, making the model better aligned with the rules of GeoGuessr and with human intuition.

---

## 🧠 How it works

1. **Dataset** – about 50k Google Street View images across ~150 countries. Each entry in `train.csv` points to an image and its country code.  
2. **Backbone** – CLIP’s ResNet50 image encoder, pretrained on 400M image–text pairs. This provides rich features for diverse environments.  
3. **Classifier** – a simple linear head that maps embeddings to country classes.  
4. **Geo-Aware Loss** – combines two components:  
   - **Similarity matrix**: derived from geodesic distances between country centroids (closer countries have higher similarity).  
   - **Class weights**: scale loss inversely with dataset frequency, so rare countries aren’t ignored.  
   - Final loss:  
     ```
     L = - Σ (w_class × sim_geo × log(p))
     ```  
5. **Evaluation** – measures both classification loss and **average geodesic error (km)** between predicted and true locations.

---

## Setup

Clone the repo and install dependencies:
```bash
git clone https://github.com/your-username/geoguessr-ai.git
cd geoguessr-ai
pip install -r requirements.txt
```

Prepare the data:
- `train.csv` with columns: `image_path,country_code`  
- `train_images/` folder containing images (organized per country)  
- `country_centroids.csv` with `country_code,lat,lon`  

---

## Training

Run:
```bash
python main.py
```

The script automatically splits the dataset into training, validation, and test sets. It applies data augmentation, learning rate scheduling, and early stopping. The best model (lowest validation loss) is saved as `best_model.pth`.

---

## Example Output
`
Epoch 12/20
Train Loss: 1.254, Val Loss: 1.201
Avg Geodesic Error: 698.4 km
LR: 0.0005
New best model saved

