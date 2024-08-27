import logging
import yaml
import torch
import torch.optim as optim
# from datasets import * 
from models import GCN1, TeacherModel
from utils import * 
from sklearn.metrics import recall_score,f1_score
import torch.nn.functional as F




def train(dataset:str,budget:int):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG,
    )
    
    logger.info('Loading config...')
    config_file = open('./config.yaml', 'r')
    config = yaml.safe_load(config_file.read())
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Loading dataset...')
    
    graph=load_from_pickle(f'../../data/train/{dataset}')

    train_graph,val_graph = train_test_split(graph=graph,ratio=0.7,edge_level_split=True,seed=0)
    
    train_graph,_,_ = relabel_graph(graph=train_graph)
    val_graph,_,_ = relabel_graph(graph=val_graph)
    train_data = preprocessing(graph=train_graph,budget=budget).to(device)
    val_data = preprocessing(graph=val_graph,budget=budget).to(device)
    
    model = TeacherModel(
        GCN1(
            in_channels=config['teacher']['in_channels'],
            hidden_channels=config['teacher']['hidden_channels'],
            out_channels=config['teacher']['out_channels']
        )
    ).to(device)

    # print(model.encoder)

    model.reset_parameters()
    

    optimizer = optim.Adam(model.parameters(), lr=config['teacher']['lr'], weight_decay=config['teacher']['weight_decay'])

    epochs = config['teacher']['epochs']

    best_val_f1= 0
    
    non_zero_indices = torch.nonzero(train_data.y).cpu()[0]
    # for epoch in range(2):
    for epoch in tqdm(range(epochs)):
        
        model.train()

        mask = torch.cat([non_zero_indices, torch.randint(0, train_data.y.size(0), (non_zero_indices.size(0),))], dim=0).to(device)

        out = model.encoder(train_data)
        # print('Out shape:',out.shape)
        out = F.log_softmax(out, dim=-1)
        # preds = out.argmax(dim=-1)
        # print(train_data.y.shape)
        loss_train = F.nll_loss(out[mask], train_data.y[mask])
        # loss_train, _ = model(train_data)
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        # acc_val, updated_weights = model.validate(data, weights.to(device))
        # acc_val = model.validate(val_data)

        out = model.encoder(val_data)
        preds = out.argmax(dim=-1)
        val_f1 = f1_score(y_true=val_data.y.cpu(), y_pred=preds.cpu())
        
        # logger.info('Epoch: [{}/{}] loss_train: {:.4f} acc_train: {:.4f} acc_val: {:.4f}'.format(epoch + 1, epochs, loss_train, acc_train, acc_val))
        if val_f1 > best_val_f1:
            # acc_best = acc_val
            best_val_f1 = val_f1
            # print(val_f1)
            torch.save(model.encoder.state_dict(), f'models/{dataset}_budget{budget}_teacher.pth')
            # torch.save({'model': model.encoder.state_dict(), 'weights': updated_weights}, config['teacher']['ckpt_path']['MVC'])
            # logger.info('Acc_best is updated to {:.4f}. Model checkpoint is saved to {}.'.format(acc_best, config['teacher']['ckpt_path']['MVC']))
            # acc_test = model.test(data)
            # logger.info('Test accuracy is {:.4f}'.format(acc_test))
        # break
        
    # logger.info('Final accuracy is {:.4f}'.format(acc_test))
            
    model.encoder.load_state_dict(torch.load(f'models/{dataset}_budget{budget}_teacher.pth'))
    print("Best F1 score:",best_val_f1)
    # print(best_val_f1)
    

    # test_graph = load_from_pickle(f'../data/test/{dataset}')
    # test_graph = load_from_pickle(f'../data/train/{dataset}')
    # graph=nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    # test_graph = nx.read_edgelist(f'../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)
    
    test_graph = load_graph(f'../../data/snap_dataset/{dataset}.txt')
    
    test_graph,_,_ = relabel_graph(graph=test_graph)
    test_data = preprocessing(graph=test_graph,budget=budget).to(device)
    model.eval()
    start = time.time()
    out = model.encoder(test_data)
    preds = out.argmax(dim=-1)
    pruned_universe = torch.nonzero(preds).squeeze().tolist()
    end= time.time()
    time_to_prune = end-start
    print('time elapsed to pruned',time_to_prune)
    
    ##################################################################

    Pg=len(pruned_universe)/test_graph.number_of_nodes()
    start = time.time()
    # objective_unpruned,queries_unpruned,solution_unpruned= greedy(graph=test_graph,budget=budget) 
    solution_unpruned = imm(graph=test_graph,seed_size=budget)


    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    subgraph = make_subgraph(test_graph,pruned_universe)
    start = time.time()
    solution_pruned = imm(graph=subgraph,seed_size=budget)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)

    objective_pruned = calculate_obj(graph=test_graph,solution=solution_pruned)
    objective_unpruned = calculate_obj(graph=test_graph,solution=solution_unpruned)
    
    
   
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of CombHelperTeacher')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    # print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'CombHelperTeacher')

    df ={      'Dataset':dataset,'Budget':budget,'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,'Ground Set': test_graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
            #   'Queries(Unpruned)': queries_unpruned, 
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
            #   'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100, 
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune

              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
    print(df)

    ###################################################################################################
    


    

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type=int, default=100, help="Budgets" )
  
    args = parser.parse_args()
    train(dataset=args.dataset,budget=args.budget)