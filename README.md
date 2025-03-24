Overview

This project explores recidivism prediction using neural networks, achieving a 10% accuracy improvement over the COMPAS algorithm. It incorporates Recursive Feature Elimination (RFE) to enhance feature selection and reliability in predictive modeling for judicial decision-making.

Technologies Used

Python

NumPy, Pandas

Scikit-Learn

TensorFlow/PyTorch

Matplotlib, Seaborn

Dataset

Due to privacy concerns, the full dataset is not included. However, a sample dataset and data preprocessing steps are provided.

Project Structure

Recidivism-Prediction/
│-- data/                     # Sample datasets (if applicable)
│-- notebooks/                 # Jupyter notebooks for analysis & training
│-- models/                    # Trained model files (if small enough)
│-- src/                       # Python scripts for preprocessing, training, and evaluation
│   │-- preprocess.py          # Data preprocessing script
│   │-- train.py               # Model training script
│   │-- evaluate.py            # Model evaluation script
│-- results/                    # Performance metrics, graphs, and visualizations
│-- .gitignore                  # Ignore unnecessary files
│-- requirements.txt            # Dependencies
│-- README.md                   # Documentation

Installation & Usage

# Clone the repository
git clone https://github.com/your-username/Recidivism-Prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the preprocessing script
python src/preprocess.py

# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py

Results

Accuracy Improvement: 10% over COMPAS

Feature Selection: Implemented Recursive Feature Elimination (RFE)

Visualization: Graphs showing model performance compared to COMPAS

Future Improvements

Test on larger datasets

Improve model explainability

Explore fairness constraints in predictions

License

This project is for research purposes. If you use any part of this code, please provide attribution.

