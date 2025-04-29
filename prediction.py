import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import json


def predict(data):
    f = open("samples_new_2.json")
    j = json.loads("\n".join(f.readlines()))
    # Example using NumPy arrays
    x = []
    y = []
    for i in j.keys():
        sample_x = []
        for k in range(0, 20):
            sample_x.append(int(j[str(i)]['board'][k]))
        sample_x.append(ord(j[str(i)]['piece']))
        sample_y = f"{j[str(i)]['movement']}:{j[str(i)]['rotation']}"
        x.append(sample_x)
        y.append(sample_y)
    X_np = np.array(x)
    y_np = np.array(y)
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(X_np)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, random_state=42)
    # Train a logistic regression model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    board = str(data[0])
    board = json.loads(board)
    sum_list = []
    for i in board:
        sum = 0
        power = len(board[0]) - 1
        for j in i:
            if (j != 0):
                sum += 2 ** power
            power -= 1
        sum_list.append(sum)
    x.append(sum_list + [ord(data[1])])
    data_np = np.array(x)
    # sc = scaler.fit_transform(data_np)
    res = model.predict(data_np)
    return res[len(res)-1]