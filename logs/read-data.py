import json
# import numpy as np 

f = open("logs/test2.txt")
l = f.readlines()
g = open("samples_new_2.json", "w")
# f = open("julia.txt")
# l = f.readlines()
# g = open("samples-julia.json", "w")

index = 0
dic = {}
sample_count = 0
while (index < len(l)):
    move_list = l[index + 2].strip()[0:-1].split(", ")
    step = 0
    rot = 0
    for move in move_list:
        if ("turn" in move):
            if (move == "turn right"):
                rot += 1
            elif (move == "turn left"):
                rot -= 1
        else:
            if (move == "right"):
                step += 1
            elif (move == "left"):
                step -= 1
    rot = rot % 4
    if (rot < 0):
        rot = 3 - (rot + 1)
    # print(move_list, step, rot)
    board = l[index + 1][0:-1]
    print(board)
    print("index: " + str(index))
    print("Max: " + str(len(l)))
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
    print(sum_list)
    dic[sample_count] = {}
    dic[sample_count]["board"] = sum_list
    dic[sample_count]["piece"] = l[index][0:-1]
    dic[sample_count]["movement"] = step
    dic[sample_count]["rotation"] = rot
    sample_count += 1
    # print([(l[index][0:-1], l[index + 1][0:-1]), (5,)])
    index += 4
g.write(json.dumps(dic))
g.close()