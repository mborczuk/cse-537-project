import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import json

f = open("logs/samples.json")
j = json.loads("\n".join(f.readlines()))

x = []
y = []
for i in range(0, 259):
    sample_x = [int(j[str(i)]['board']), ord(j[str(i)]['piece'])]
    sample_y = f"{j[str(i)]['movement']}:{j[str(i)]['rotation']}"
    x.append(sample_x)
    y.append(sample_y)
X_np = np.array(x)
y_np = np.array(y)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X_np)
print(scaled_data)

X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_np, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_train)

print(y_train)
print(predictions)