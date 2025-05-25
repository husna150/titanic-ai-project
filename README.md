# Titanic Survival Prediction Project
This project explores machine learning models (Logistic Regression, Decision Tree, Random Forest, Stacked Model) and deep learning models (Artificial Neural Network and Recurrent Neural Network) to predict passenger survival on the Titanic. The dataset contains information about individual passengers, and our goal is to build models that can accurately predict whether a passenger survived or not.
## Dataset Preparation

The dataset was preprocessed with the following steps:
- Dropped rows with missing values in 'Age' and 'Embarked' columns
- Encoded categorical variables:
  - 'Sex': male → 0, female → 1
  - 'Embarked': S → 0, C → 1, Q → 2
- Selected features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
- Normalized features for neural network models

## Models Implemented

We evaluated several machine learning and deep learning models:

### Traditional ML Models
1. **Logistic Regression**: Baseline model
2. **Decision Tree**: Simple tree-based model
3. **Random Forest**: Ensemble of decision trees

### Advanced/Ensemble Models
1. **Stacked Model**: Combination of Random Forest, Decision Tree, and Logistic Regression

### Deep Learning Models
1. **Neural Network**: Feedforward ANN with dropout layers
2. **RNN**: Recurrent Neural Network for sequential data analysis

## Results

### Model Accuracy Comparison
| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.7972   |
| Decision Tree        | 0.6923   |
| Random Forest        | 0.7622   |
| Stacked Model        | 0.8042   ||
| Neural Network       | 0.8112   |
| RNN                  | 0.8042   |


### ROC Curve Performance
| Model                | AUC Score |
|----------------------|-----------|
| Logistic Regression  | 0.83      |
| Decision Tree        | 0.69      |
| Random Forest        | 0.84      |
| Stacked Model        | 0.85      |
| RNN                  | 0.82      |

### Feature Importance (Random Forest)
The most important features for prediction were:
1. Sex (0.25)
2. Fare (0.15)
3. Age (0.13)
4. Pclass (0.12)
5. SibSp (0.08)
6. Parch (0.07)
7. Embarked (0.05)

## Deep Learning Details

### Neural Network Architecture
- Input layer: 7 features
- Hidden layers: 
  - Dense(64, relu) with Dropout(0.3)
  - Dense(32, relu) with Dropout(0.3)
- Output layer: Dense(1, sigmoid)
- Optimizer: Adam
- Epochs: 100
- Batch size: 16
- Final accuracy: 81.82%

### RNN Architecture
- Input shape: (None, 1, 7)  
- RNN layer: SimpleRNN(32)
- Dense layers:
  - Dense(16, relu)
  - Dense(1, sigmoid)
- Optimizer: Adam
- Epochs: 100
- Batch size: 16
- Final accuracy: 80.42%

## Key Findings
- The Stacked Model performed best among traditional ML approaches (80.42% accuracy)
- Neural Network achieved the highest overall accuracy (81.12%)
- RNN model showed competitive performance (80.42%) despite not being sequential data
- Sex was the most important predictive feature
- The Stacked Model had the highest AUC score (0.85)

## How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the Jupyter notebooks or Python scripts:
   - `traditional_ml.py` for basic models
   - `deep_learning.py` for ANN and RNN models
   - `visualization.ipynb` for result analysis

## Requirements
- Python 3.7+
- pandas
- scikit-learn
- tensorflow/keras
- matplotlib


## Future Work
- Experiment with more advanced RNN architectures (LSTM, GRU)
- Try transformer-based approaches
- Incorporate additional feature engineering
- Implement hyperparameter tuning for all models
- Explore attention mechanisms for the RNN
## 👋 About Me

I'm a machine learning enthusiast with hands-on experience in AI models like ANN, RNN, and stacking. 

🔍 Looking for freelance or paid AI projects.

📧 Reach me at: husnabsmaths@gmail.com 
