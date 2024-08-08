import logging
import yaml
import torch
import torch.optim as optim
from datasets import *
from models import GCN1, StudentModel
from utils import *

from sklearn.metrics import recall_score,f1_score




def train(dataset,budget):
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Loading dataset...')
    graph=load_from_pickle(f'../data/train/{dataset}')

    train_graph,val_graph = train_test_split(graph=graph,ratio=0.7,edge_level_split=True,seed=0)
    
    train_graph,_,_ = relabel_graph(graph=train_graph)
    val_graph,_,_ = relabel_graph(graph=val_graph)
    train_data = preprocessing(graph=train_graph,budget=budget).to(device)
    val_data = preprocessing(graph=val_graph,budget=budget).to(device)


   
    logger.info('Loading teacher model...')
    # checkpoint = torch.load(config['teacher']['ckpt_path']['MVC'])
    encoder_t = GCN1(
        in_channels=config['teacher']['in_channels'],
        hidden_channels=config['teacher']['hidden_channels'],
        out_channels=config['teacher']['out_channels']
    )

    encoder_t.load_state_dict(torch.load(f'models/{dataset}_budget{budget}_teacher.pth',map_location=device))
    # encoder_t.load_state_dict(torch.load(f'data/{dataset}_teacher.pth',map_location=device))
    logger.info('Teacher GNN backbone is GraphSAGE')
    logger.info('In channels: {}'.format(config['teacher']['in_channels']))
    logger.info('Hidden channels: {}'.format(config['teacher']['hidden_channels']))
    logger.info('Out channels: {}'.format(config['teacher']['out_channels']))
    
    logger.info('Loading weights...')
    # weights = checkpoint['weights'].detach()
    
    logger.info('Get student model')
    encoder_s = GCN1(
        in_channels=config['student']['in_channels'],
        hidden_channels=config['student']['hidden_channels'],
        out_channels=config['student']['out_channels']
    )
    logger.info('Student GNN backbone is GraphSAGE')
    logger.info('In channels: {}'.format(config['student']['in_channels']))
    logger.info('Hidden channels: {}'.format(config['student']['hidden_channels']))
    logger.info('Out channels: {}'.format(config['student']['out_channels']))
    
    model = StudentModel(
        encoder_t=encoder_t,
        encoder_s=encoder_s,
        T=config['student']['T'],
        alpha=config['student']['alpha'],
        beta=config['student']['beta'],
        boosting=config['student']['boosting'],
        num_class=config['student']['out_channels']
    ).to(device)
    
    logger.info('Reseting model parameters...')
    model.reset_parameters()
    
    logger.info('Get optimizer')
    optimizer = optim.Adam(model.parameters(), lr=config['student']['lr'], weight_decay=config['student']['weight_decay'])
    logger.info('Learning rate: {}'.format(config['student']['lr']))
    logger.info('Weight decay: {}'.format(config['student']['weight_decay']))
    
    logger.info('Start training...')
    acc_best = 0.0
    epochs = config['student']['epochs']

    # weights
    out = encoder_t(train_data)
    acc_train = int((out.argmax(dim=-1) == train_data.y).sum()) / len(train_data.y)
    error = 1 - acc_train
    alpha = 0.5 * np.log((1 - error) / error)
    weights = degree (train_data.edge_index[0]).view(-1, 1)
    updated_weights = weights / weights.sum()
    updated_weights[out.argmax(dim=-1) == train_data.y] *= np.exp(0 - alpha)
    updated_weights[out.argmax(dim=-1) != train_data.y] *= np.exp(alpha)
    # updated_weights *= torch.exp(weights) # why ?
    updated_weights = updated_weights.to(device)

    best_val_f1= 0

    non_zero_indices = torch.nonzero(train_data.y).cpu()[0]
    for epoch in tqdm(range(epochs)):
        
        model.train()
        mask = torch.cat([non_zero_indices, torch.randint(0, train_data.y.size(0), (non_zero_indices.size(0),))], dim=0).to(device)
        loss_train, acc_train = model(train_data, updated_weights,mask=mask)
        loss_train.backward()
        optimizer.step()

        model.eval()

        out = model.encoder_s(val_data)
        preds = out.argmax(dim=-1)
        val_f1 = f1_score(y_true=val_data.y.cpu(), y_pred=preds.cpu())
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # print("Best F1 score:",best_val_f1)
            torch.save(model.encoder_s.state_dict(), f'models/{dataset}_budget{budget}_student.pth')

    
    print("Best F1 score:",best_val_f1)
    print('Performance of Student Model')
    model.encoder_s.load_state_dict(torch.load(f'models/{dataset}_budget{budget}_student.pth'))


    # test_graph = load_from_pickle(f'../data/test/{dataset}')
    test_graph = nx.read_edgelist(f'../data/snap_dataset/{dataset}.txt', create_using=nx.Graph(), nodetype=int)
    # test_graph = load_from_pickle(f'../data/train/{dataset}')
    test_graph,_,_ = relabel_graph(graph=test_graph)
    test_data = preprocessing(graph=test_graph,budget=budget).to(device)
    model.eval()

    out = model.encoder_s(test_data)
    preds = out.argmax(dim=-1)

    solution = torch.nonzero(preds).squeeze().tolist()

    # print('Solution:',torch.nonzero(preds).tolist())

    pruned_solution,pruned_queries= greedy(graph= test_graph,budget=budget,ground_set=solution)
    greedy_solution,unpruned_queries = greedy(graph=test_graph,budget=budget)

    Pg = len(solution)/test_graph.number_of_nodes()
    greedy_objective = calculate_cover(test_graph,greedy_solution)
    ratio = calculate_cover(test_graph,pruned_solution)/ greedy_objective

    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(solution))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    print('Queries:',round(pruned_queries/unpruned_queries,4)*100)

    

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type=int, default=10, help="Budgets" )
  
    args = parser.parse_args()
    train(dataset=args.dataset,budget=args.budget)