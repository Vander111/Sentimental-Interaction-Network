import torch

def opt_SGD(model,lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return optimizer

def opt_RMS(model,lr):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 1e-7)
    return optimizer,scheduler

def get_opt(opt_name,model,lr,args=None):
    if opt_name == 'SGD':
        opt = opt_SGD(model,lr)
        sch = None
    elif opt_name == 'RMS':
        opt,sch = opt_RMS(model,lr)
    else:
        print('opt name not found')
    return opt,sch
