# Tumor Classification
## Project Description
This project focuses on classifying tumors as malignant or benign using machine learning techniques. The model is designed for participation in a hackathon where the primary goal is to enhance prediction accuracy.

## Data Structure
The training and testing data contain numerous medical features such as texture, perimeter, and area of tumors. The data is split into training and testing sets.

## Technologies
The project utilizes the following technologies and libraries:
* Python
* Pandas and NumPy for data handling
* Scikit-learn for building and evaluating machine learning models
* TQDM for displaying progress bars during training
  
## Workflow
1. Data Loading: The script loads data from CSV files and combines them into a single dataset.
2. Data Preprocessing: Unnecessary columns are removed, missing values are filled, and features are scaled.
3. Meta-feature Generation: Using ensemble methods, meta-features are generated based on predictions from several models.
4. Model Training: The model is trained on training data, including both original and meta-features.
5. Model Evaluation: The model's performance is assessed on test data.
6. Prediction for Test Set: Predictions for the test set are generated and saved in CSV format for submission to the competition.
   
## Model Output Results
Predictions of the model for the test dataset have been saved in the file submission.csv. The model was evaluated using the accuracy metric, which achieved a result of 0.92280 on the test dataset.

Project Launch
Step-by-step instructions for project launch:
1. Clone the repository: 
`git clone https://github.com/DatRush/Hackathon-Skillfactory.git`
2. Install the required libraries: 
`pip install -r requirements.txt`
3. To run the: 
`python ensemble.py`
The results will be saved in CSV format.