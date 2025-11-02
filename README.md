# DeforNet

**DeforNet** is a deep learning project that uses satellite imagery to automatically classify land types — helping monitor deforestation and promote sustainable land management.

---

## 🛰️ Overview

Think of ForestWatch as a **digital park ranger** in the sky.  
It scans satellite images, recognizes what kind of land is in each tile (like **Forest**, **Urban**, **Water**, or **Cleared Land**), and flags any changes over time — helping detect illegal logging, wildfires, and other deforestation events.

### 🔍 How It Works
1. **Input**: A small satellite image tile.  
2. **Model (CNN)**: A convolutional neural network trained on labeled satellite images.  
   - Learns visual patterns, textures, and colors that define each land type.  
3. **Output**: A single class label (e.g., “Forest”, “Water”, “Urban”).  

When run on a time series of images, ForestWatch can **automatically detect areas that change from forest to cleared land**, providing real-time alerts for potential deforestation.

---

## 🌍 Why It Matters

- **🌲 Large-Scale Monitoring** — Scan millions of satellite images in hours, not months.  
- **⚡ Early Warning System** — Detect new deforestation events almost immediately.  
- **📍 Pinpoint Accuracy** — Get exact GPS coordinates of affected regions.  
- **📊 Research & Accountability** — Provide data for governments, NGOs, and scientists to track land-use change and climate impacts.

By automating image classification, ForestWatch acts as a **watchdog for the planet’s forests**.

---

## 🧠 Model & Dataset

- **Model**: Convolutional Neural Network (CNN) built using TensorFlow/Keras (or PyTorch).  
- **Dataset**: [*Trees in Satellite Imagery*](https://www.kaggle.com/datasets) (from Kaggle).  
- **Classes**: Forest, Urban, Water, Cleared Land, Agriculture, etc.  

The model learns to distinguish these categories based on textures, colors, and shapes visible in satellite tiles.

---

## ⚙️ Project Structure

