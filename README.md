# 🌿 Plant Disease Detection with EfficientNetV2B0

An end-to-end deep learning project designed to identify **38 different classes** of plant diseases from leaf images with high precision.

---

## 🚀 Features
* **Core Architecture:** Fine-tuned **EfficientNetV2B0** architecture for high-efficiency feature extraction.
* **High Accuracy:** Achieved **96% accuracy** on the comprehensive PlantVillage dataset.
* **Robustness:** Optimized with **Class Weights** and strategic fine-tuning to differentiate between visually similar diseases.
* **Interactive Web App:** Real-time diagnostics via a user-friendly **Streamlit** interface.

---

## 🛠️ The Pipeline

The project follows a structured computer vision workflow to ensure reliable and scalable predictions:

1.  **Preprocessing:** Standardizing input images to $128 \times 128 \times 3$ and applying EfficientNet-specific scaling and normalization using `preprocess_input`.
2.  **Feature Extraction:** Utilizing the pre-trained EfficientNetV2B0 base to identify complex spatial patterns and leaf textures.
3.  **Classification:** A custom head consisting of **Global Average Pooling**, a **Batch Normalization** layer, and **Dropout (0.3)** for regularization, ending in a 38-way Softmax output.
4.  **Deployment:** Model serialization to `.keras` format, integrated into a Streamlit backend for instant image-to-diagnosis results.



---

## 🧠 Computer Vision Concepts Applied

This project utilizes several advanced CV techniques to ensure high accuracy and robust generalization:

* **Transfer Learning:** Leveraged pre-trained weights from ImageNet to give the model a "head start" in recognizing fundamental shapes, edges, and textures.
* **Fine-Tuning:** Strategically unfroze the top 30 layers of the base model. This allowed the network to adapt its high-level filters to the specific "micro-textures" of plant pathologies (e.g., fungal spots vs. bacterial lesions).
* **Data Augmentation:** Implemented a pipeline of geometric (rotation, zoom, horizontal flip) and photometric (brightness adjustment) transformations to reduce overfitting and improve performance across varied lighting conditions.
* **Global Average Pooling (GAP):** Used GAP instead of a traditional Flatten layer to reduce the total number of parameters and minimize spatial variance, making the model more robust.
* **Class Balancing:** Addressed dataset imbalance using **Cost-Sensitive Learning** (Class Weights), ensuring the model maintains high recall even for rare disease categories.



---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Handling:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

---

## 📊 Performance
The model was fine-tuned by unfreezing the top layers of the EfficientNet base, significantly improving the recall for similar-looking diseases. Through the use of a **Learning Rate Scheduler (ReduceLROnPlateau)**, the model successfully converged to a stable 96% accuracy.

