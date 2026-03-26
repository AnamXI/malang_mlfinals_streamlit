import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv('Customer_Segmentation_Data.csv')
df.columns = df.columns.str.strip().str.lower()

le_gender = LabelEncoder()
le_category = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['preferred_category'] = le_category.fit_transform(df['preferred_category'])

features = ['age', 'gender', 'income', 'membership_years', 'purchase_frequency', 'preferred_category']
X = df[features]
y = df['spending_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'segmentation_model.joblib')
joblib.dump(le_gender, 'le_gender.joblib')
joblib.dump(le_category, 'le_category.joblib')
joblib.dump(features, 'feature_names.joblib')
joblib.dump(model.feature_importances_, 'feature_importances.joblib')
# Save the averages so the UI knows what a "Normal" customer looks like
joblib.dump(df[features].mean(), 'feature_means.joblib')

print("Success: Model trained and feature priorities saved!")