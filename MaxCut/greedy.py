from argparse import ArgumentParser
from utils import *
import pandas as pd
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt

    

def get_gains(graph:nx.Graph,ground_set=None):
    if ground_set is None:

        gains={node:graph.degree(node) for node in graph.nodes()}
    else:
        print('A ground set has been given')

        gains={node:graph.degree(node) for node in ground_set}
        print('Size of the ground set = ',len(gains))

    
    return gains

    
def gain_adjustment(graph:nx.Graph,gains:dict,selected_element:list,spins):

    gains[selected_element]=-gains[selected_element]

    for neighbor in graph.neighbors(selected_element):

        if neighbor in gains:
            gains[neighbor]+=(2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element]=1-spins[selected_element]



def greedy(graph:nx.Graph,budget:list,ground_set=None):
    
    number_of_queries = 0

    gains = get_gains(graph,ground_set)
    
    solution=[]
    # uncovered=defaultdict(lambda: True)
    spins={node:1 for node in graph.nodes()}
    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)

        obj_val += gains[selected_element]
        
        gain_adjustment(graph,gains,selected_element,spins)
    print('Objective value =', obj_val)
    print('Number of queries =',number_of_queries)

    return obj_val,number_of_queries,solution

        











