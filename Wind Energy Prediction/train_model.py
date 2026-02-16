import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("T1.csv")

df.rename(columns={
    "Wind Speed (m/s)": "wind_speed",
    "Wind Direction (°)": "wind_direction",
    "Theoretical_Power_Curve (KWh)": "theoretical_power",
    "LV ActivePower (kW)": "actual_power"
}, inplace=True)

df = df[["wind_speed","wind_direction","theoretical_power","actual_power"]]

sns.heatmap(df.corr(),annot=True)
plt.show()

X = df[["wind_speed","theoretical_power"]]
y = df["actual_power"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("R2 Score:",r2_score(y_test,y_pred))

pickle.dump(model,open("wind_model.pkl","wb"))

print("Model saved successfully")