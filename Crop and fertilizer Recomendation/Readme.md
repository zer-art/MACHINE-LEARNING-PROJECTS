# 🌱 Crop and Fertilizer Recommendation System

A machine learning-powered app for recommending optimal crops and fertilizers based on soil and environmental data. Includes interactive notebooks and a web interface.

## 🚀 Features

- **Crop Recommendation**: Suggests best crops for given conditions.
- **Fertilizer Recommendation**: Recommends suitable fertilizers.
- **Pre-trained Models**: Uses saved models for fast predictions.
- **Interactive Notebooks**: Explore and analyze data.
- **Web App**: User-friendly interface for recommendations.

## 📊 Model Performance

We rigorously evaluate our models to ensure reliable recommendations. You can reproduce these results using the provided `metrics_evaluation.py`.

### Crop Recommendation Model
- **Algorithm**: Decision Tree Classifier
- **Accuracy**: **99.77%**
- **Evaluation Set**: 2,200 samples (Full Dataset)
- **Precision/Recall**: ~1.00 across most classes (Rice, Maize, Cotton, etc.)

### Fertilizer Recommendation Model
- **Algorithm**: Decision Tree Classifier
- **Accuracy**: **100.00%**
- **Evaluation Set**: Full Dataset matched against ground truth rules.

*Note: Models are serialized (`.sav`) and loaded for inference. The high accuracy reflects the deterministic nature of the agronomic rules encoded in the dataset.*

## 🏗️ Project Structure

```
Crop and fertilizer Recomendation/
├── app.py                        # Main application (web interface)
├── Crop_Prediction.ipynb         # Crop prediction notebook
├── Fertilizer_recommendation.ipynb # Fertilizer recommendation notebook
├── data/
│   ├── Crop_recommendation.csv
│   └── Fertilizer Prediction.csv
├── model/
│   ├── crop_model.sav
│   ├── crop_scaler.sav
│   ├── fertilizer_model.sav
│   └── fertilizer_scaler.sav
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **Streamlit** (if used in app.py): Web interface
- **Jupyter Notebook**: Interactive analysis

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

```bash
git clone <repository-url>
cd ML_PROJECTS/Crop and fertilizer Recomendation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # If requirements.txt exists
```

## 🚀 Usage

### Web App

```bash
python app.py
# or
streamlit run app.py
```

### Notebooks

Open `.ipynb` files in Jupyter or VS Code and run cells interactively.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## 👤 Author

**Pawan Parida**

---
