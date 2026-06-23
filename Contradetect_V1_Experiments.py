import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Load the dataset
dataset_path = "Sythetic Contradetect V1.xlsx" 
df = pd.read_excel(dataset_path) 

#----WORD COUNTS----
# Calculate sentence lengths (word count)
#df['sentence_1_length'] = df['Premise'].apply(lambda x: len(x.split()))
#df['sentence_2_length'] = df['Hypothesis'].apply(lambda x: len(x.split()))

# Plot the distribution of sentence lengths
#plt.figure(figsize=(10, 5))
#sns.histplot(df['sentence_1_length'], kde=True, color='blue', label='Premise', bins=10)
#sns.histplot(df['sentence_2_length'], kde=True, color='red', label='Hypothesis', bins=10)
#plt.title('Sentence Length Distribution')
#plt.xlabel('Sentence Length (words)')
#plt.ylabel('Frequency')
#plt.legend()
#plt.show()

#----Actual Important code----
# First split into training and test sets train 90:train 10:test 
train_data, test_data = train_test_split(df, test_size=0.1, stratify=df['Relationship'])

# Then split the training data into training and validation sets 89:train 9:validate
train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data['Relationship'])

# Print sizes of splits
print(f"Training Set Size: {len(train_data)}")
print(f"Validation Set Size: {len(val_data)}")
print(f"Test Set Size: {len(test_data)}\n\n")

# Extract features and labels from training, validation, and test data
X_train = train_data['Premise'] + " " + train_data['Hypothesis']  
y_train = train_data['Relationship']
X_val = val_data['Premise'] + " " + val_data['Hypothesis']
y_val = val_data['Relationship']
X_test = test_data['Premise'] + " " + test_data['Hypothesis']
y_test = test_data['Relationship']

# Convert text data to numerical features using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english' , max_features=3575)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Encode labels 
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

#--Define Print Metrics--
def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

#----RF TEST----
# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42,n_jobs=-1)

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"Random Forest Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nRandom Forest Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nRandom Forest Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----RF BALANCED TEST----
# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42,n_jobs=-1, class_weight="balanced")

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"Balanced Random Forest Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nBalanced Random Forest Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nBalanced Random Forest Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----LR TEST----
rf=LogisticRegression(random_state=42)

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"Logistic Regression Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nLogistic Regression Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nLogistic Regression Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----LR BALANCED TEST----
rf=LogisticRegression(random_state=42, class_weight="balanced")

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"Logistic Regression Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nLogistic Regression Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nLogistic Regression Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----SVC TEST----
rf=LinearSVC(random_state=42)

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"LinearSVC Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nLinearSVC Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nLinearSVC Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----SVC BALANCED TEST----
rf=LinearSVC(random_state=42, class_weight="balanced")

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"LinearSVC Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nLinearSVC Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nLinearSVC Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----GBC TEST----
rf=GradientBoostingClassifier(random_state=42)

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"GBC Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nGBC Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nGBC Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")

#----MNB TEST----
rf=MultinomialNB()

# Perform Cross-Validation (CV)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(rf, X_train_tfidf, y_train_encoded, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

# Print the cross-validation results
print(f"MNB Cross-validation results:")
for score in scoring:
    print(f"{score.capitalize()}: Mean = {cv_results[f'test_{score}'].mean():.4f}, Std = {cv_results[f'test_{score}'].std():.4f}")

# Train the Random Forest model on the full training data
rf.fit(X_train_tfidf, y_train_encoded)

# Make predictions on the validation and test sets
y_val_pred = rf.predict(X_val_tfidf)
y_test_pred = rf.predict(X_test_tfidf)

#Print Metrics
# Print metrics for validation data
print("\nMNB Validation Metrics:")
print_metrics(y_val_encoded, y_val_pred)

# Print metrics for test data
print("\nMNB Test Metrics:")
print_metrics(y_test_encoded, y_test_pred)
print("\n")


