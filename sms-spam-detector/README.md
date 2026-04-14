# SMS Spam Detection Pipeline

### Objective
To build a Machine Learning classification model capable of distinguishing between legitimate human text messages ("Ham") and unsolicited garbage ("Spam").

### Data Preprocessing & Architecture
* **Dataset:** The classic SMS Spam Collection dataset.
* **Cleaning:** Ghost columns were removed, and remaining features were mapped to standard `label` and `message` identifiers.
* **Validation Strategy:** Implemented a stratified 80/20 Train/Test split to preserve the natural imbalance of the Ham/Spam ratio and prevent data leakage.
* **Vectorization:** Utilized Scikit-Learn's `CountVectorizer` within a strictly ordered Pipeline to automatically lowercase text, strip punctuation, remove English stop-words, and convert the raw strings into a mathematical sparse matrix (Bag of Words) strictly on the training data.

### Model Comparison
Three algorithms were evaluated using the strict pipeline architecture:
1. **Multinomial Naive Bayes (Baseline):** Chosen for its mathematical efficiency with sparse matrices and word counts.
2. **Logistic Regression:** Tested as a conservative alternative focused on minimizing False Positives.
3. **K-Nearest Neighbors (KNN):** Tested to demonstrate the algorithm's vulnerability to the Curse of Dimensionality on sparse text matrices.

### Results & Conclusion
**The winning model is Multinomial Naive Bayes.** While Logistic Regression technically achieved 0 False Positives, its recall on actual spam was too low (it allowed 26 spam messages through the filter). The K-Nearest Neighbors model failed entirely, missing 65% of all spam due to its inability to calculate distances accurately across thousands of zero-filled columns.

The Naive Bayes model proved to be the most balanced and effective algorithm. It achieved an overall accuracy of 98%, caught 92% of all hidden spam in the test data, and only misclassified 6 legitimate messages out of nearly 1,000. 

### Deliverables Included
* Full Jupyter Notebook with execution history
* Scikit-Learn Pipelines for automated vectorization and prediction
* Model Comparison via Confusion Matrices and Classification Reports
* Saved `.pkl` artifact of the final trained Naive Bayes pipeline
* 
### Repository Structure
* `/data`: Contains the raw `spam.csv` dataset.
* `/notebooks`: Contains the `spam_classifier.ipynb` Jupyter Notebook detailing the exploratory data analysis, cleaning, and model training workflow.
* `/models`: Contains the exported `best_spam_pipeline.pkl` artifact for deployment.
* `README.md`: Project documentation.

### How to Use the Model
To load the pre-trained model and make predictions on new text data, use the `joblib` library:
```python
import joblib
pipeline = joblib.load('models/best_spam_pipeline.pkl')
prediction = pipeline.predict(["Your text message here"])