import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
import joblib
df=pd.read_csv("custumer_churn.csv")
#print(df.info())
#print(df.isnull().sum())
print(df.columns)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna( df["TotalCharges"].median())
#df.dropna(inplace=True)
#print(df["TotalCharges"].isnull().sum())
df = pd.get_dummies(df, drop_first=True)
#print(df.columns.tolist())
#print(df.corr()["Churn"].sort_values(ascending=False))
selected_features = [
    'TotalCharges',
    'tenure',
    'MonthlyCharges',
    'InternetService_Fiber optic',
    'InternetService_No',
    'PaymentMethod_Electronic check',
    'Contract_Two year',
    'OnlineSecurity_Yes',
    'gender_Male',
    'PaperlessBilling_Yes',
    'Contract_One year'
]

X = df[selected_features]
y = df["Churn"]
print (X)
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2,random_state=42)
#print(X_train.shape)
#print(X_test.shape)
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
#y_pred=model.predict(X_test)
#print(classification_report(y_test, y_pred))
#importance = pd.DataFrame({
 #   "Feature": X.columns,
 #})

#importance = importance.sort_values(
 ###)

#print(importance.head(10))
joblib.dump(model,"customer_churn_model.pkl")
joblib.dump(X.columns.tolist(),"features.pkl")

print("model successfully saved") 
