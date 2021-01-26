import time
import para
from utils import *
from trainer.trainer import train
from trainer.trainer_gcn import train as traingcn
import torch
from logger import log,update_best_acc
import sys


def set_card(id):
    GPU = id
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def main():

    declines = [10, 20]
    
    if len(sys.argv) > 1:
        trainparm_file = sys.argv[1]
        para_train = load_json(trainparm_file)
    else:
        project_root = get_project_root()
        para_path = project_root + 'train_parameter.json'
        para_train = load_json(para_path)

    GPU = para_train["train_data"]["card id"]
    device = set_card(GPU)

    for n in range(0, 10):
        para_train["train_data"]["lr"] = para_train["train_data"]["lr"] + 0.005 * n
        for m in range(0, len(declines)):
            para_train["train_data"]["decline"] = declines[m]
            para_train["train_data"]["save_root"] = save_root
            model_data = tuple(para_train["model_data"].values())
            train_data = tuple(para_train["train_data"].values())
            set_data = tuple(para_train["set_data"].values())
            epoch_num = para_train["train_data"]["epoch_num"]
            train_set(para_train, model_data, train_data, set_data, device)


def train_set(para_train,model_data, train_data, set_data,device):

    mode, backbone, fc_init_flag,\
    node_inputsize, node_size, class_num = model_data

    lr,loss_name,decline,save_fre,epoch_num,batch_size,opt_name,save_root,card = train_data

    data_name, edge_name, node_name, train_mode,transform_flag, val_flag = set_data

    time_train_start = time.strftime("%Y%m%d_%H%M", time.localtime())
    date2log = re.split('_',time_train_start)[0]
    time2log = re.split('_',time_train_start)[1]
    sodir = save_root + backbone + '/' + str(data_name) + '/' + time_train_start + '/'
    if not os.path.exists(sodir):
        os.makedirs(sodir)
    print('-----------------------------')
    print(time_train_start)
    print(sodir)
    print('-----------------------------')
    print('saving traing parameter...')
    save_json(sodir+'train_parameter.json',para_train)
    print('loading data...')
    datas, edge_path, train_splits, label_dict, best_acc_log = para.get_data_info(data_name, edge_name, node_name, train_mode, class_num)
    train_loader, test_loader, val_loader, nums = para.get_dataloader(train_splits, datas, transform_flag, batch_size,
                                                                 val_flag,
                                                                 label_dict)
    train_num, test_num, val_num = nums
    print('loading model...')
    model = para.get_model(mode, backbone, fc_init_flag, batch_size, node_inputsize, node_size,
                      class_num, device)
    model.to(device)
    print('making graph...')

    graph = para.mk_graph(edge_path)
    graph.edata['rela'] = graph.edata['rela'].to(device)

    loss = para.get_loss(loss_name)
    optimizer, scheduler = para.get_opt(opt_name, model, lr)

    data = model, epoch_num, lr, decline, loss, optimizer, train_loader,\
           test_loader, val_loader, sodir, save_fre, train_num, test_num,val_num,\
           graph, batch_size, scheduler
    print('start training...')
    print('best acc before:',best_acc_log)

    acc,best_model_path = train(data)

    if acc>best_acc_log:
        update_best_acc(data_name, class_num, mode, acc, best_model_path)

    content2log = date2log,time2log,lr,decline,fusion_weights,node_name,edge_name,acc
    log(content2log,save_root + backbone + '/' + str(data_name) + '/log.txt')

if __name__ == '__main__':
    main()