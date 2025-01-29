# # Cell 1: Import Libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# %matplotlib
# inline
# plt.style.use('seaborn')
#
# # Cell 2: Load and Prepare Data
# # Load the data
# # Replace this with your actual data file path
# dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
# np.random.seed(42)
# n = len(dates)
#
# df = pd.DataFrame({
#     'Date': dates,
#     'Adj Close': np.random.normal(100, 20, n).cumsum(),
#     'Close': np.random.normal(100, 20, n).cumsum(),
#     'High': np.random.normal(102, 20, n).cumsum(),
#     'Low': np.random.normal(98, 20, n).cumsum(),
#     'Open': np.random.normal(100, 20, n).cumsum(),
#     'Volume': np.random.normal(1000000, 200000, n).abs()
# })
#
# # Display first few rows
# print("First few rows of the dataset:")
# display(df.head())
#
# # Cell 3: Feature Engineering
# # Create target variable (1 if price went up, 0 if down)
# df['Target'] = (df['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
#
# # Create technical indicators
# df['Returns'] = df['Adj Close'].pct_change()
# df['Volume_Change'] = df['Volume'].pct_change()
# df['High_Low_Range'] = df['High'] - df['Low']
# df['Open_Close_Range'] = df['Close'] - df['Open']
#
# # Create moving averages
# df['MA5'] = df['Adj Close'].rolling(window=5).mean()
# df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#
# # Create RSI
# delta = df['Adj Close'].diff()
# gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
# loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# rs = gain / loss
# df['RSI'] = 100 - (100 / (1 + rs))
#
# # Drop NaN values
# df = df.dropna()
#
# # Select features for model
# features = ['Returns', 'Volume_Change', 'High_Low_Range',
#             'Open_Close_Range', 'MA5', 'MA20', 'RSI']
#
# print("\nFeatures created:")
# for feature in features:
#     print(f"- {feature}")
#
# # Cell 4: Exploratory Data Analysis
# # Basic statistics
# print("\nBasic Statistics:")
# display(df[features + ['Target']].describe())
#
# # Correlation matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(df[features + ['Target']].corr(), annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()
#
# # Target distribution
# plt.figure(figsize=(8, 6))
# df['Target'].value_counts().plot(kind='bar')
# plt.title('Distribution of Target Variable')
# plt.xlabel('Price Movement (0: Down, 1: Up)')
# plt.ylabel('Count')
# plt.show()
#
# # Feature distributions
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# axes = axes.ravel()
#
# for idx, col in enumerate(features):
#     if idx < len(axes):
#         sns.histplot(data=df, x=col, ax=axes[idx])
#         axes[idx].set_title(f'Distribution of {col}')
#
# plt.tight_layout()
# plt.show()
#
# # Cell 5: Data Split
# X = df[features]
# y = df['Target']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# print("Training set shape:", X_train.shape)
# print("Testing set shape:", X_test.shape)
#
# # Cell 6: Train Models
# # Gini Model
# dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
# dt_gini.fit(X_train, y_train)
#
# # Entropy Model
# dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
# dt_entropy.fit(X_train, y_train)
#
#
# # Cell 7: Model Evaluation
# def evaluate_model(model, X_test, y_test, model_name):
#     # Make predictions
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
#
#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#
#     print(f"\n{model_name} Results:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#
#     return y_pred, y_pred_proba
#
#
# # Evaluate both models
# y_pred_gini, y_pred_proba_gini = evaluate_model(dt_gini, X_test, y_test, "Gini Model")
# y_pred_entropy, y_pred_proba_entropy = evaluate_model(dt_entropy, X_test, y_test, "Entropy Model")
#
#
# # Cell 8: Feature Importance
# def plot_feature_importance(model, features, model_name):
#     importances = pd.DataFrame({
#         'feature': features,
#         'importance': model.feature_importances_
#     }).sort_values('importance', ascending=True)
#
#     plt.figure(figsize=(10, 6))
#     plt.barh(importances['feature'], importances['importance'])
#     plt.title(f'Feature Importance ({model_name})')
#     plt.xlabel('Importance')
#     plt.show()
#
#
# plot_feature_importance(dt_gini, features, "Gini")
# plot_feature_importance(dt_entropy, features, "Entropy")
#
#
# # Cell 9: ROC Curves
# def plot_roc_curves(models, X_test, y_test):
#     plt.figure(figsize=(10, 6))
#
#     for name, model in models.items():
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
#         fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#         roc_auc = auc(fpr, tpr)
#
#         plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
#
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves: Gini vs Entropy')
#     plt.legend()
#     plt.show()
#
#
# plot_roc_curves({'Gini': dt_gini, 'Entropy': dt_entropy}, X_test, y_test)
#
#
# # Cell 10: Confusion Matrices
# def plot_confusion_matrices(y_test, y_pred_gini, y_pred_entropy):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
#     sns.heatmap(confusion_matrix(y_test, y_pred_gini), annot=True, fmt='d', ax=ax1)
#     ax1.set_title('Confusion Matrix (Gini)')
#     ax1.set_xlabel('Predicted')
#     ax1.set_ylabel('Actual')
#
#     sns.heatmap(confusion_matrix(y_test, y_pred_entropy), annot=True, fmt='d', ax=ax2)
#     ax2.set_title('Confusion Matrix (Entropy)')
#     ax2.set_xlabel('Predicted')
#     ax2.set_ylabel('Actual')
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_confusion_matrices(y_test, y_pred_gini, y_pred_entropy)
#
# # Cell 11: Visualize Decision Trees
# plt.figure(figsize=(20, 10))
# plot_tree(dt_gini, feature_names=features, filled=True, max_depth=3)
# plt.title('Decision Tree (Gini) - Limited to Depth 3 for Visibility')
# plt.show()
#
# plt.figure(figsize=(20, 10))
# plot_tree(dt_entropy, feature_names=features, filled=True, max_depth=3)
# plt.title('Decision Tree (Entropy) - Limited to Depth 3 for Visibility')
# plt.show()
#
# # Cell 12: Results and Conclusion
# print("Model Comparison Summary:")
# print("-" * 50)
#
# # Compare feature importances
# gini_importances = pd.DataFrame({
#     'feature': features,
#     'gini_importance': dt_gini.feature_importances_
# })
#
# entropy_importances = pd.DataFrame({
#     'feature': features,
#     'entropy_importance': dt_entropy.feature_importances_
# })
#
# comparison = gini_importances.merge(entropy_importances, on='feature')
# comparison = comparison.sort_values('gini_importance', ascending=False)
#
# print("\nFeature Importance Comparison:")
# display(comparison)
#
# print("\nKey Findings:")
# print("1. Model Performance:")
# print("   - Compare accuracy, precision, recall, and F1 scores above")
# print("2. Feature Importance:")
# print(f"   - Most important feature (Gini): {comparison.iloc[0]['feature']}")
# print(f"   - Most important feature (Entropy): {comparison.iloc[0]['feature']}")
# print("3. ROC Analysis:")
# print("   - Compare AUC scores from ROC curves above")
# print("4. Model Complexity:")
# print(f"   - Gini tree depth: {dt_gini.get_depth()}")
# print(f"   - Entropy tree depth: {dt_entropy.get_depth()}")