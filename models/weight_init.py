
def fc_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)
