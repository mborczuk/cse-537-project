import json
import numpy as np 

f = open("logs\\test.txt")
l = f.readlines()
g = open("samples.json", "w")

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
    print(move_list, step, rot)
    board = l[index + 1][0:-1]
    board = json.loads(board)
    power = len(board[0]) * len(board) - 1
    print(power)
    sum = 0
    for i in board:
        for j in i:
            if (j != 0):
                sum += 2 ** power
            power -= 1
    print(sum)
    print(len(str(sum)))
    dic[sample_count] = {}
    dic[sample_count]["board"] = sum
    dic[sample_count]["piece"] = l[index][0:-1]
    dic[sample_count]["movement"] = step
    dic[sample_count]["rotation"] = rot
    sample_count += 1
    # print([(l[index][0:-1], l[index + 1][0:-1]), (5,)])
    index += 4
g.write(json.dumps(dic))
g.close()