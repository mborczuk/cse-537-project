import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import json


def predict(data):
    f = open("samples_new_2.json")
    j = json.loads("\n".join(f.readlines()))
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
    # scaler = MinMaxScaler()

    X_1_np = np.array(x_1)
    y_1_np = np.array(y_1)
    # scaled_data_1 = scaler.fit_transform(X_1_np)
    # X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(scaled_data_1, y_1_np, random_state=42)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1_np, y_1_np, random_state=42)
    model_1 =  RandomForestClassifier()
    model_1.fit(X_train_1, y_train_1)

    X_2_np = np.array(x_2)
    y_2_np = np.array(y_2)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2_np, y_2_np, random_state=42)
    model_2 =  RandomForestClassifier()
    model_2.fit(X_train_2, y_train_2)

    X_3_np = np.array(x_3)
    y_3_np = np.array(y_3)
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3_np, y_3_np, random_state=42)
    model_3 =  RandomForestClassifier()
    model_3.fit(X_train_3, y_train_3)

    X_4_np = np.array(x_4)
    y_4_np = np.array(y_4)
    X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4_np, y_4_np, random_state=42)
    model_4 =  RandomForestClassifier()
    model_4.fit(X_train_4, y_train_4)

    X_5_np = np.array(x_5)
    y_5_np = np.array(y_5)
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5_np, y_5_np, random_state=42)
    model_5 = RandomForestClassifier()
    model_5.fit(X_train_5, y_train_5)

    X_6_np = np.array(x_6)
    y_6_np = np.array(y_6)
    X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_6_np, y_6_np, random_state=42)
    model_6 =  RandomForestClassifier()
    model_6.fit(X_train_6, y_train_6)

    X_7_np = np.array(x_7)
    y_7_np = np.array(y_7)
    X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X_7_np, y_7_np, random_state=42)
    model_7 =  RandomForestClassifier()
    model_7.fit(X_train_7, y_train_7)


    board = str(data[0])
    board = json.loads(board)
    # power = len(board[0]) * len(board) - 1
    sum_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    row = 0;
    for i in board:
        loc = 0;
        for j in i:
            if (j != 0 and sum_list[loc] < j):
                sum_list[loc] = 20-row;
            loc+=1;
        row+=1;
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
