import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- 1. Load the dataset ---
try:
    df = pd.read_csv('/Users/shashwat/Desktop/SIDEMEN copy/data/synthetic_fuel_11000_rules_noisy.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'synthetic_fuel_11000_balanced.csv' not found. Please ensure the file is in the correct directory.")
    df = None # Set df to None if file not found


# Check if df was loaded successfully
if df is not None:
    # --- 2. Data Preparation and Preprocessing ---
    # Safe fillna without FutureWarnings
    df['previous_fuel_level'] = df['previous_fuel_level'].fillna(0)
    df['fuel_diff'] = df['fuel_diff'].fillna(0)
    # Removed the line accessing 'data_quality_flag' as it doesn't exist in the loaded dataframe

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Define features and target
    X = df.drop(columns=['timestamp', 'eventType']) # Removed 'data_quality_flag' from drop list
    y = df['eventType']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify categorical feature indices for CatBoost
    categorical_features_indices = [X.columns.get_loc(col) for col in ['ignitionStatus', 'isOverSpeed']]

    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 4. Calculate Class Weights ---
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    readable_weights = {label_encoder.inverse_transform([cls])[0]: f"{weight:.2f}"
                        for cls, weight in class_weights_dict.items()}
    print("\nCalculated Class Weights:")
    print(readable_weights)

    # --- 5. Hyperparameter Tuning with RandomizedSearchCV ---
    base_model = CatBoostClassifier(
        loss_function='MultiClass',
        cat_features=categorical_features_indices,
        class_weights=class_weights_dict,
        random_seed=42,
        verbose=0
    )

    param_distributions = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1, 0.2],
        'iterations': [500, 1000, 1500]
    }

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='f1_weighted',
        cv=3,
        n_jobs=-1,
        verbose=3,
        random_state=42
    )

    print("\n--- Starting Hyperparameter Tuning ---")
    random_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning Complete ---")
    print("Best parameters found: ", random_search.best_params_)
    print(f"Best F1-Weighted score on CV: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_

    # --- 6. Save the best model and label encoder ---
    model_filename = 'catboost_model_random_tuned.cbm'
    best_model.save_model(model_filename)
    print(f"\nBest model saved successfully to '{model_filename}'")

    label_encoder_filename = 'label_encoder.pkl'
    joblib.dump(label_encoder, label_encoder_filename)
    print(f"Label encoder saved successfully to '{label_encoder_filename}'")

    # Save the training columns for later use in the API
    training_columns_filename = 'training_columns.pkl'
    joblib.dump(X.columns.tolist(), training_columns_filename)
    print(f"Training columns saved successfully to '{training_columns_filename}'")


    # --- 7. Evaluate the best model on the test set ---
    y_pred_encoded = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_encoded)
    print(f"\nBest Model Accuracy on Test Set: {accuracy:.4f}")

    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded.flatten())
    y_test_labels = label_encoder.inverse_transform(y_test)

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (Tuned Model with Class Weights)')
    plt.ylabel('Actual Event Type')
    plt.xlabel('Predicted Event Type')
    plt.show()

    # --- 8. Load the saved model and verify prediction ---
    print("\n--- Loading Saved Model for Verification ---")
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")

    # Load the label encoder for verification
    loaded_label_encoder = joblib.load(label_encoder_filename)
    print(f"Label encoder '{label_encoder_filename}' loaded successfully.")


    sample_data = X_test.iloc[[15]]
    actual_label_encoded = y_test[15]
    actual_label = loaded_label_encoder.inverse_transform([actual_label_encoded])[0]

    sample_prediction_encoded = loaded_model.predict(sample_data)
    predicted_label = loaded_label_encoder.inverse_transform(sample_prediction_encoded.flatten())[0]

    print("\n--- Single Prediction Example ---")
    print("Sample Data Input:")
    print(sample_data)
    print(f"\nActual Event Type: {actual_label}")
    print(f"Predicted Event Type: {predicted_label}")
else:
    print("\nDataFrame 'df' was not created due to the file not being found. Skipping subsequent steps.")