import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import json

f = open("logs/samples.json")
j = json.loads("\n".join(f.readlines()))

x = []
y = []
for i in range(0, 259):
    print(int(j[str(i)]['board'][0]))
    sample_x = []
    for k in range(0, 20):
        sample_x.append(int(j[str(i)]['board'][k]))
    sample_x.append(ord(j[str(i)]['piece']))
    sample_y = f"{j[str(i)]['movement']}:{j[str(i)]['rotation']}"
    x.append(sample_x)
    y.append(sample_y)
X_np = np.array(x)
y_np = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(y_test)
print(predictions)

count = 0
for i in range(0, len(y_test)):
    if (y_test[i] == predictions[i]):
        count += 1

print(count)
print(len(y_test))
print(count / len(y_test))