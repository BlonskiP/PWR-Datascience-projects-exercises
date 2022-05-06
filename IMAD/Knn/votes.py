import sys

def majority_vote(neighbors, knn, row):

    classes = [row.to_list()[-1] for row in neighbors]
    prediction = max(set(classes), key=classes.count)
    return prediction

def distance_weighted_vote(neighbors, knn, row):
    classes_set = set([row.to_list()[-1] for row in neighbors])
    distance_values = []
    votes = []
    for index , class_item in enumerate(classes_set):
        distance_values.append(0)
        votes.append(0)
        for neighbour in neighbors:
            if neighbour.to_list()[-1] == class_item:
                votes[index] += 1
                distance_values[index] += knn.distance_func(neighbour[0:-1],row)
        if distance_values[index] != 0:
            distance_values[index]=1/distance_values[index]
        else:
            distance_values[index] = 1/(float('-inf'))
        votes[index] = votes[index]*distance_values[index]
    res = list(zip(classes_set,votes))
    #print(res)
    prediction = max( res, key= lambda x:x[1])
    return prediction[0]

def farest_distance_weighted_vote(neighbors, knn, row):
    classes_set = set([row.to_list()[-1] for row in neighbors])
    distance_values = []
    max_dist=[]
    min_dist=[]
    votes = []
    for index , class_item in enumerate(classes_set):
        distance_values.append(0)
        votes.append(0)
        max_dist.append(0)
        min_dist.append(float('inf'))
        for neighbour in neighbors:
            if neighbour.to_list()[-1] == class_item:
                votes[index] += 1
                temp_dist = knn.distance_func(neighbour[0:-1],row)
                if temp_dist > max_dist[index]:
                    max_dist[index]=temp_dist
                if temp_dist < min_dist[index]:
                    min_dist[index]=temp_dist

                weight = abs(max_dist[index]-min_dist[index])
                if weight > distance_values[index]:
                    distance_values[index]=weight
        if distance_values[index] != 0:
            distance_values[index]=1/distance_values[index]
        else:
            distance_values[index] = 1/(float('-inf'))
        votes[index] = votes[index]*distance_values[index]
    res = list(zip(classes_set,votes))
    #print(res)
    prediction = max( res, key= lambda x:x[1])
    return prediction[0]