Here is a well-structured **README.md** file for your project:  

---

## **Speech Emotion Recognition (SER) using Machine Learning & Deep Learning**  

### **📌 Project Overview**  
This project aims to classify emotions from speech using **Machine Learning (Decision Tree)** and **Deep Learning (CNN)** models. It utilizes **MFCC features** extracted from **RAVDESS and TESS** datasets to train models for recognizing emotions like **neutral, happy, sad, angry, fearful, disgust, surprised, and calm**.

---

### **📂 Folder Structure**  
```
📦 PROJ
 ┣ 📂 data                       # Raw dataset files (if applicable)
 ┣ 📂 dataset_features           # Processed dataset stored as joblib files
 ┃ ┣ 📜 X.joblib                 # Feature set (MFCCs)
 ┃ ┣ 📜 y.joblib                 # Labels corresponding to features
 ┣ 📂 models                     # Saved trained models
 ┃ ┣ 📜 cnn_model.h5             # Trained CNN model
 ┃ ┣ 📜 decision_tree.pkl        # Trained Decision Tree model
 ┣ 📂 src                        # Core source code
 ┃ ┣ 📜 data_loader.py           # Loads dataset features
 ┃ ┣ 📜 evaluation.py            # Evaluates models
 ┃ ┣ 📜 model_saving.py          # Saves and loads models
 ┃ ┣ 📜 model_training.py        # Trains Decision Tree & CNN models
 ┃ ┣ 📜 visualize_results.py     # Plots accuracy/loss graphs
 ┣ 📜 config.py                  # Project configurations
 ┣ 📜 create_features.py         # Extracts MFCC features & saves joblib files
 ┣ 📜 input.py                   # Handles input processing
 ┣ 📜 main.py                    # Runs the full pipeline (training & evaluation)
 ┣ 📜 pred.py                    # Predicts emotion from new audio
 ┣ 📜 requirements.txt           # Required Python libraries
 ┣ 📜 sample_audio.wav           # Sample audio file for testing
 ┣ 📜 tess_pipeline_for_data.py  # Prepares dataset folders
 ┗ 📜 README.md                  # Project documentation
```

---

### **📊 Dataset Information**  
The dataset consists of **5,252 speech samples** from:  

- **[RAVDESS](https://zenodo.org/record/1188976)**: **2,452 samples** (speech & song) recorded by **24 actors** expressing 8 emotions.  
- **[TESS](https://tspace.library.utoronto.ca/handle/1807/24487)**: **2,800 samples**, recorded by **two actresses** speaking target words in 7 emotions.  

Each sample is converted into **40-dimensional MFCC features** for model training.

---

### **⚙️ Installation & Setup**  
#### **1️⃣ Clone the repository**  
```bash
git clone <repo-link>
cd PROJ
```
#### **2️⃣ Install dependencies**  
```bash
pip install -r requirements.txt
```
#### **3️⃣ Run the main pipeline**  
```bash
python main.py
```
#### **4️⃣ Predict emotion from an audio file**  
```bash
python pred.py --file sample_audio.wav
```

---

### **🖥️ Model Training & Evaluation**  
- **Decision Tree**: Baseline ML model, **67% accuracy**  
- **CNN**: Deep learning model, **84% accuracy**  

To retrain models:  
```bash
python main.py --train
```

---

### **📊 Results & Observations**  
✅ **CNN outperforms Decision Tree**, learning deep hierarchical speech features.  
✅ **Best classified emotion:** Neutral (94% Precision).  
✅ **Most misclassified emotion:** Surprised (79% Recall).  
✅ **Potential improvements:** Hybrid CNN-LSTM model, dataset augmentation, transformer-based architectures.

---

### **📩 Contributors**  
- **Bhushan Gunjal**
- **Pravin Patrike**
- **Varsha Rai**

For any questions, feel free to reach out! 🚀🔥  

---

This **README** is clean, concise, and follows best practices. Let me know if you'd like any modifications! 🚀📄