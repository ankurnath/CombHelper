import logging
import yaml
import torch
import torch.optim as optim
from datasets import *
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
    
    graph=load_from_pickle(f'../data/train/{dataset}')

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
    for epoch in range(epochs):
        
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
            print(val_f1)
            torch.save(model.state_dict(), f'data/{dataset}_teacher.pth')
            # torch.save({'model': model.encoder.state_dict(), 'weights': updated_weights}, config['teacher']['ckpt_path']['MVC'])
            # logger.info('Acc_best is updated to {:.4f}. Model checkpoint is saved to {}.'.format(acc_best, config['teacher']['ckpt_path']['MVC']))
            # acc_test = model.test(data)
            # logger.info('Test accuracy is {:.4f}'.format(acc_test))
        # break
        
    # logger.info('Final accuracy is {:.4f}'.format(acc_test))
            
    model.load_state_dict(torch.load(f'data/{dataset}_teacher.pth'))


    # test_graph = load_from_pickle(f'../data/test/{dataset}')
    test_graph = load_from_pickle(f'../data/train/{dataset}')
    test_graph,_,_ = relabel_graph(graph=test_graph)
    test_data = preprocessing(graph=test_graph,budget=budget).to(device)
    model.eval()

    out = model.encoder(test_data)
    preds = out.argmax(dim=-1)

    solution = torch.nonzero(preds).cpu()[0].tolist()

    print(solution)

    pruned_solution,_= greedy(graph= test_graph,budget=budget,ground_set=solution)
    greedy_solution,_ = greedy(graph=test_graph,budget=budget)
    print('Ratio:',calculate_cover(test_graph,pruned_solution)/calculate_cover(test_graph,greedy_solution))


    

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type=int, default=10, help="Budgets" )
  
    args = parser.parse_args()
    train(dataset=args.dataset,budget=args.budget)