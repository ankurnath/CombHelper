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
        print('A ground set has been given')
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

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]==0:
            print('All elements are already covered')
            break
        solution.append(selected_element)
        # print(gains[selected_element])
        gain_adjustment(graph,gains,selected_element,uncovered)

    print('Number of queries:',number_of_queries)


    return solution,number_of_queries

        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", nargs='+', type=int, help="Budgets" )
  
    args = parser.parse_args()

    # file_path=f'../../data/test/{args.dataset}'

    # file_path=f'../../data/s/{args.dataset}'
    # graph=load_from_pickle(file_path)
    load_graph_file_path=f'../../data/snap_dataset/{args.dataset}.txt'
    graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)


    covers = []

    # solution,_ = greedy(graph=graph,budget=graph.number_of_nodes())
    solution,_ = greedy(graph=graph,budget=100)

    covers = [calculate_cover(graph,solution[:i+1]) for i in range(len(solution))]
    # cover = 0
    # for i in range(1,graph.number_of_nodes()+1):
    
    #     if cover != graph.number_of_nodes():
    #         solution,_=greedy(graph=graph,budget=i)
    #         cover=calculate_cover(graph,solution)
    #         print(cover)
    #         covers.append(cover)
    #     else:
    #         break

    plt.figure(dpi=200)
    plt.plot (range(1,len(covers)+1),covers)
    plt.xlabel('Budget')
    plt.ylabel('Cover')
    plt.title (f'{args.dataset}')
    plt.show()
    
    # df=defaultdict(list)

    # for budget in args.budgets:

    #     # solution,_=greedy(graph=graph,budget=budget)

    #     # solution,_ = greedy (graph=graph,budget=graph.number_of_nodes())

    #     subgraph =make_subgraph(graph,solution)
    #     Pv=1-subgraph.number_of_nodes()/graph.number_of_nodes()
    #     Pe=1-subgraph.number_of_edges()/graph.number_of_edges()

    #     cover=calculate_cover(graph,solution)

    #     df['Budget'].append(budget)
    #     df['Pv'].append(Pv)
    #     df['Pe'].append(Pe)
    #     df['Objective Value'].append(cover)
    #     df['Objective Value (Ratio)'].append(cover/graph.number_of_nodes())
    #     df['Solution'].append(solution)
        
        
    # df=pd.DataFrame(df)
    # print(df)
    # save_folder=f'data/{args.dataset}'
    # file_path=os.path.join(save_folder,'Greedy')
    # os.makedirs(save_folder,exist_ok=True)
    # save_to_pickle(df,file_path)








