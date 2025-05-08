import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json


def predict(data):
    f = open("logs/samples-julia-outline.json")
    j = json.loads("\n".join(f.readlines()))
    # f2 = open("data_new_1.json")
    # j2 = json.loads("\n".join(f2.readlines()))
    # Example using NumPy arrays
    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    x_3 = []
    y_3 = []
    x_4 = []
    y_4 = []
    x_5 = []
    y_5 = []
    x_6 = []
    y_6 = []
    x_7 = []
    y_7 = []

    for i in j.keys():
        sample_x = j[str(i)]['board']
        # for k in range(0, 20):
        #     sample_x.append(int(j[str(i)]['board'][k]))
        sample_y = f"{j[str(i)]['movement']}:{j[str(i)]['rotation']}"
        if(j[str(i)]['piece'] == "L"):
            x_1.append(sample_x)
            y_1.append(sample_y)
        elif(j[str(i)]['piece'] == "S"):
            x_2.append(sample_x)
            y_2.append(sample_y)
        elif(j[str(i)]['piece'] == "J"):
            x_3.append(sample_x)
            y_3.append(sample_y)
        elif(j[str(i)]['piece'] == "I"):
            x_4.append(sample_x)
            y_4.append(sample_y)
        elif(j[str(i)]['piece'] == "O"):
            x_5.append(sample_x)
            y_5.append(sample_y)
        elif(j[str(i)]['piece'] == "Z"):
            x_6.append(sample_x)
            y_6.append(sample_y)
        elif(j[str(i)]['piece'] == "T"):
            x_7.append(sample_x)
            y_7.append(sample_y)
    # for i in j2.keys():
    #     sample_x = j2[str(i)]['board']
    #     # for k in range(0, 20):
    #     #     sample_x.append(int(j[str(i)]['board'][k]))
    #     sample_y = f"{j2[str(i)]['movement']}:{j2[str(i)]['rotation']}"
    #     if(j2[str(i)]['piece'] == "J"):
    #         x_3.append(sample_x)
    #         y_3.append(sample_y)
    #     elif(j2[str(i)]['piece'] == "T"):
    #         x_7.append(sample_x)
    #         y_7.append(sample_y)
    scaler = MinMaxScaler()

    X_1_np = np.array(x_1)
    y_1_np = np.array(y_1)
    scaled_data_1 = scaler.fit_transform(X_1_np)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(scaled_data_1, y_1_np, random_state=42)
    model_1 = RandomForestClassifier(n_estimators=40)
    model_1.fit(X_train_1, y_train_1)
    y_pred_1 = model_1.predict(X_test_1);
    accuracy = accuracy_score(y_test_1, y_pred_1)
    print(f"Accuracy 1: {accuracy*100:.2f}%")

    X_2_np = np.array(x_2)
    y_2_np = np.array(y_2)
    scaled_data_2 = scaler.fit_transform(X_2_np)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(scaled_data_2, y_2_np, random_state=42)
    model_2 = RandomForestClassifier(n_estimators=43)
    model_2.fit(X_train_2, y_train_2)
    y_pred_2 = model_2.predict(X_test_2);
    accuracy = accuracy_score(y_test_2, y_pred_2)
    print(f"Accuracy 2: {accuracy*100:.2f}%")

    X_3_np = np.array(x_3)
    y_3_np = np.array(y_3)
    scaled_data_3 = scaler.fit_transform(X_3_np)
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(scaled_data_3, y_3_np, random_state=42)
    model_3 = RandomForestClassifier(n_estimators=50)
    model_3.fit(X_train_3, y_train_3)
    y_pred_3 = model_3.predict(X_test_3);
    accuracy = accuracy_score(y_test_3, y_pred_3)
    print(f"Accuracy 3: {accuracy*100:.2f}%")

    X_4_np = np.array(x_4)
    y_4_np = np.array(y_4)
    scaled_data_4 = scaler.fit_transform(X_4_np)
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(scaled_data_4, y_4_np, random_state=42)
    model_4 = RandomForestClassifier(n_estimators=50)
    model_4.fit(X_train_4, y_train_4)
    y_pred_4 = model_4.predict(X_test_4);
    accuracy = accuracy_score(y_test_4, y_pred_4)
    print(f"Accuracy 4: {accuracy*100:.2f}%")

    X_5_np = np.array(x_5)
    y_5_np = np.array(y_5)
    scaled_data_5 = scaler.fit_transform(X_5_np)
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(scaled_data_5, y_5_np, random_state=42)
    model_5 = RandomForestClassifier(n_estimators=40)
    model_5.fit(X_train_5, y_train_5)
    y_pred_5 = model_5.predict(X_test_5);
    accuracy = accuracy_score(y_test_5, y_pred_5)
    print(f"Accuracy 5: {accuracy*100:.2f}%")

    X_6_np = np.array(x_6)
    y_6_np = np.array(y_6)
    scaled_data_6 = scaler.fit_transform(X_6_np)
    X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(scaled_data_6, y_6_np, random_state=42)
    model_6 = RandomForestClassifier(n_estimators=40)
    model_6.fit(X_train_6, y_train_6)
    y_pred_6 = model_6.predict(X_test_6);
    accuracy = accuracy_score(y_test_6, y_pred_6)
    print(f"Accuracy 6: {accuracy*100:.2f}%")

    X_7_np = np.array(x_7)
    y_7_np = np.array(y_7)
    scaled_data_7 = scaler.fit_transform(X_7_np)
    X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(scaled_data_7, y_7_np, random_state=42)
    model_7 = RandomForestClassifier(n_estimators=40)
    model_7.fit(X_train_7, y_train_7)
    y_pred_7 = model_1.predict(X_test_7);
    accuracy = accuracy_score(y_test_7, y_pred_7)
    print(f"Accuracy 7: {accuracy*100:.2f}%")


    board = str(data[0])
    board = json.loads(board)
    power = len(board[0]) * len(board) - 1

    sum_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    row = 0;
    for i in board:
        loc = 0;
        for j in i:
            if (j != 0 and sum_list[loc] < j):
                sum_list[loc] = 20-row;
            loc+=1;
        row+=1;



    # sum_list = []

    # for i in board:
    #     sum = 0
    #     power = len(board[0]) - 1
    #     # print(power)
    #     for j in i:
    #         if (j != 0):
    #             sum += 2 ** power
    #         power -= 1
    #     # print(sum)
    #     sum_list.append(sum)

    x_1.append(sum_list)
    data_np = np.array(x_1);
    res = model_1.predict(data_np);
    if(str(data[1]) == "L"):
        x_1.append(0);
    elif(str(data[1]) == "S"):
        x_2.append(sum_list)
        data_np = np.array(x_2);
        res = model_2.predict(data_np);
    elif(str(data[1]) == "J"):
        x_3.append(sum_list)
        data_np = np.array(x_3);        
        res = model_3.predict(data_np);
    elif(str(data[1]) == "I"):
        x_4.append(sum_list)
        data_np = np.array(x_4);       
        res = model_4.predict(data_np);
    elif(str(data[1])== "O"):
        x_5.append(sum_list)
        data_np = np.array(x_5);       
        res = model_5.predict(data_np);
    elif(str(data[1]) == "Z"):
        x_6.append(sum_list)
        data_np = np.array(x_6);       
        res = model_6.predict(data_np);
    elif(str(data[1]) == "T"):
        x_7.append(sum_list)
        data_np = np.array(x_7);       
        res = model_7.predict(data_np);
    return res[len(res)-1];
