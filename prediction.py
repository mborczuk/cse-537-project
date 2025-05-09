import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import json


def predict(data, classifier_mode, dataset_mode):
    f = open("logs/samples-julia.json") # regular dataset
    if (dataset_mode == 1): 
        f = open("logs/samples-julia-outline.json") # outline dataset
    j = json.loads("\n".join(f.readlines()))

    x = []
    y = []
    for i in j.keys():
        sample_x = []
        for k in range(0, 10):
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
    model = None
    if (classifier_mode == 0):
        model = RandomForestClassifier()
    elif (classifier_mode == 1):
        model = KNeighborsClassifier(n_neighbors=37)
    elif (classifier_mode == 2):
        model = MLPClassifier(hidden_layer_sizes=(100, 100))
    else:
        raise Exception("Invalid model selected.") 
    model.fit(X_train, y_train)

    board = str(data[0])
    board = json.loads(board)
    power = len(board[0]) * len(board) - 1

    if (classifier_mode == 1): # outline datasets
        sum_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        row = 0;
        for i in board:
            loc = 0;
            for j in i:
                if (j != 0 and sum_list[loc] < j):
                    sum_list[loc] = 20-row;
                loc+=1;
            row+=1;
    if (classifier_mode == 0): # regular dataset
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

    res = model.predict(data_np)
    return res[len(res)-1]