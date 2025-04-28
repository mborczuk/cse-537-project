import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import json


def predict(data):
    f = open("logs/samples.json")
    j = json.loads("\n".join(f.readlines()))
    # Example using NumPy arrays
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
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_np, random_state=42)
    # Train a logistic regression model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    board = str(data[0])
    board = json.loads(board)
    power = len(board[0]) * len(board) - 1
    sum = 0
    for i in board:
        for j in i:
            if (j != 0):
                sum += 2 ** power
            power -= 1
    x.append([sum, ord(data[1])])
    data_np = np.array(x);
    sc = scaler.fit_transform(data_np)
    res = model.predict(sc);
    return res[len(res)-1];