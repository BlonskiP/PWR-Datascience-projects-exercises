from math import sqrt


def euclidean(x1, x2): #euclidean distance between series
    distance = 0.0
    x1_size = len(x1)
    for i in range(x1_size):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)


def manhattan(x1, x2): #this aproch has O(nlogn)
    return sum(abs(a-b) for a,b in zip(x1,x2))
