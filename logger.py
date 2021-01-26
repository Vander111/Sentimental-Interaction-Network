from utils import *

def save_or_delete(run_epoch_num,sodir,threoshold=5):
    if run_epoch_num <= threoshold:
        os.remove(sodir)

def get_best_record(data_info,data_name,class_num,mode):
    if class_num == 2:
        class_part = 'class_2'
    else:
        class_part = 'class_m'
    if mode == 1:
        mode_part = 'ori_gcn'
    else:
        mode_part = 'fore_ori_gcn'
    best_acc = data_info[data_name]["best_model"][class_part][mode_part]['acc']
    return best_acc

def update_best_acc(data_name,class_num,mode,acc,model_path):
    project_root = get_project_root()
    data_info = load_json(project_root + 'data_path.json')
    if class_num == 2:
        class_part = 'class_2'
    else:
        class_part = 'class_m'
    if mode == 1:
        mode_part = 'ori_gcn'
    else:
        mode_part = 'fore_ori_gcn'
    data_info[data_name]["best_model"][class_part][mode_part]['acc'] = acc
    data_info[data_name]["best_model"][class_part][mode_part]['path'] = model_path
    save_json(project_root + 'data_path.json',data_info)

def log(content,log_path):

    if not os.path.isfile(log_path):
        f = open(log_path,'w')
        f.write('date,time,lr,decline,weights,node_feature,edges,GPU,acc')
        f.write('\n')
        f.close()
    towrite = ''
    for item in list(content):
        towrite += str(item) + ','

    f = open(log_path,'a+')
    f.write(towrite)
    f.write('\n')
    f.close()
