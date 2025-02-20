# -Real-Time-Quality-Validation-for-Streaming-Data-using-AI
Automated real-time quality validation system using AI to detect anomalies in streaming data. This project uses Machine Learning models to analyze and validate incoming streaming data for quality issues.
 Installation & Setup
1️⃣ Clone the Repository:

bash
Copy
Edit
git clone https://github.com/your-username/Real-Time-Quality-Validation-for-Streaming-Data.git
cd Real-Time-Quality-Validation-for-Streaming-Data
2️⃣ Install Required Libraries:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open credit_risk_analysis.ipynb to explore the dataset and model.

🛠 How to Use?
📌 Steps to Run the Code:
1️⃣ Load the dataset using Pandas
2️⃣ Perform Exploratory Data Analysis (EDA)
3️⃣ Preprocess the data (Handle missing values, Encode categorical features)
4️⃣ Train Machine Learning models (Logistic Regression, Random Forest, XGBoost)
5️⃣ Evaluate models using Confusion Matrix & Classification Report

Run the following command in Jupyter Notebook to train the model:

python
Copy
Edit
python train_model.py
📊 Model Performance
🔹 Logistic Regression Performance
makefile
Copy
Edit
Accuracy: 71%  
Precision: 70%  
Recall: 71%  
F1-Score: 70%  
🔹 Random Forest Performance
makefile
Copy
Edit
Accuracy: 80%  
Precision: 80%  
Recall: 80%  
F1-Score: 80%  
🔹 Random Forest performed better than Logistic Regression.

📌 Confusion Matrix Visualization:

🚀 Future Improvements
✅ Optimize hyperparameters using GridSearchCV
✅ Use LSTM for real-time streaming predictions
✅ Deploy as an API using FastAPI or Flask

🤝 Contributors
Nishark123
Team Member 4

License
This project is licensed under the MIT License.
