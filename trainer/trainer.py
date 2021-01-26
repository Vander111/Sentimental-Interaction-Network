import torch
import os
import time

def train(data):
    model, epoch_num, lr, lr_decay, loss_func, optimizer, train_loader, test_loader, val_loader, sodir, save_fre, \
    train_num, test_num, val_num, graph, batchsize, scheduler = data
    logpath = sodir + 'log.txt'
    time_cur = time.asctime(time.localtime(time.time()))
    if os.path.isfile(logpath):
        f = open(logpath, 'a+')
    else:
        f = open(logpath, 'w')
    f.write(time_cur)
    f.write('\n')
    f.close()

    best_acc = 0

    for epoch in range(epoch_num):
        print('-----------------------------')
        print('epoch:', epoch)
        train_loss = 0.0
        train_acc = 0.0
        n = 0
        model.train()

        for trainData, trainLabel in train_loader:

            img_names,img, features = trainData
            features = features.cuda()
            trainLabel = trainLabel.cuda()

            optimizer.zero_grad()
            out = model(img, features, graph)

            loss = loss_func(out, trainLabel)
            loss.backward()
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == trainLabel).sum()
            train_acc += train_correct.item()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (train_num),
                                                       float(train_acc) / float(train_num)))
        log = str(epoch) + ',' + str(train_loss / train_num) + ',' + str(
            float(train_acc) / float(train_num)) + ','

        if (epoch + 1) % save_fre == 0:
            savepath = sodir + 'resgraph.pth'
            print('Model save', epoch + 1, sodir)
            torch.save(model, savepath)

        if (epoch + 1) % lr_decay == 0:
            lr = lr / 10
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        model.eval()
        test_acc = 0.0
        for testData, testLabel in test_loader:
            img_names,img, features = testData
            features = features.cuda()
            testLabel = testLabel.cuda()
            out = model(img, features, graph)
            pred = torch.max(out, 1)[1]
            test_correct = (pred == testLabel).sum()
            test_acc += test_correct.item()
        test_acc = test_acc / (test_num)

        if test_acc > best_acc:
            best_acc = test_acc
            print('test_acc:', test_acc, 'best_acc', best_acc)
            savepath = sodir + 'best.pth'
            print('Best model save:', epoch + 1, sodir)
            torch.save(model, savepath)
        else:
            print('test_acc:', test_acc, 'best_acc', best_acc)

        log += str(test_acc) + ',' + str(best_acc)
        f = open(logpath, 'a+')
        f.write(log)
        f.write("\n")
        f.close()
    return best_acc,sodir + 'best.pth'