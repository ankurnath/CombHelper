from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt
#TO
def select_variable(gains):
    sum_gain = sum(gains.values())
    if sum_gain==0:
        return None
    else:
        prob_dist=[gains[key]/sum_gain for key in gains]
        element=np.random.choice([key for key in gains], p=prob_dist)
        return element
    

def get_gains(graph,ground_set):
    if ground_set is None:

        gains={node:graph.degree(node)+1 for node in graph.nodes()}
    else:
        # print('A ground set has been given and Size:',len(ground_set))
        gains={node:graph.degree(node)+1 for node in ground_set}
        # print('Size of ground set',len(gains))
    return gains

    
def gain_adjustment(graph,gains,selected_element,uncovered):

    # print('Gains:',gains[selected_element])

    # uncovered[selected_element]=False
    # for neighbor in graph.neighbors(selected_element):
    #     uncovered[neighbor]=False

    # for node in gains:
    #     gains[node]= 1 if uncovered[node] else 0
    #     for neighbor in graph.neighbors(node):
    #         if uncovered[neighbor]:
    #             gains[node]+=1
            

    if uncovered[selected_element]:
        gains[selected_element]-=1
        uncovered[selected_element]=False
        for neighbor in graph.neighbors(selected_element):
            if neighbor in gains and gains[neighbor]>0:
                gains[neighbor]-=1

    for neighbor in graph.neighbors(selected_element):
        if uncovered[neighbor]:
            uncovered[neighbor]=False
            
            if neighbor in gains:
                gains[neighbor]-=1
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                if neighbor_of_neighbor  in gains:
                    gains[neighbor_of_neighbor ]-=1


    assert gains[selected_element] == 0


def prob_greedy(graph,budget,ground_set=None,delta=0):


    gains = get_gains(graph,ground_set)

    solution = []
    uncovered = defaultdict(lambda: True)


    for _ in range(budget):

        selected_element = select_variable(gains)

        if selected_element is None or gains[selected_element]<delta:
            break
        solution.append(selected_element)
        gain_adjustment(graph,gains,selected_element,uncovered)
    return solution



def greedy(graph,budget,ground_set=None):
    
    number_of_queries = 0

    gains = get_gains(graph,ground_set)
    


    solution=[]
    uncovered=defaultdict(lambda: True)

    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)
        

        obj_val += gains[selected_element]
        gain_adjustment(graph,gains,selected_element,uncovered)

    print('Objective value =', obj_val)
    print('Number of queries =',number_of_queries)

    


    return solution,number_of_queries

        











