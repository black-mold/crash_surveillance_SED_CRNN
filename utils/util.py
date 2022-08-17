
import torch
import os

from nnet.CRNN import *



def save(EPOCH, net, optimizer, TRAIN_LOSS, VAL_LOSS, PATH):

    save_path = "{}/CRNN_epoch{}.pth".format(PATH, EPOCH) # 수정해야함.

    torch.save({
            'epoch': EPOCH,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': TRAIN_LOSS,
            'val_loss' : VAL_LOSS
            }, save_path)

def save_best_model(EPOCH, net, optimizer, TRAIN_LOSS, VAL_LOSS, f1score, PATH):

    save_path = "{}/bestmodel.pth".format(PATH) # 수정해야함.

    torch.save({
            'epoch': EPOCH,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': TRAIN_LOSS,
            'val_loss' : VAL_LOSS,
            'val_f1' : f1score
            }, save_path)
    
    print("Epoch %d 에서 best model 저장\n-------------------------------" % (EPOCH))

def save_bestf1_model(EPOCH, net, optimizer, TRAIN_LOSS, VAL_LOSS, f1score, PATH):

    save_path = "{}/bestf1model.pth".format(PATH) # 수정해야함.

    torch.save({
            'epoch': EPOCH,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': TRAIN_LOSS,
            'val_loss' : VAL_LOSS,
            'val_f1' : f1score
            }, save_path)
    
    print("Epoch %d 에서 bestf1 model 저장\n-------------------------------" % (EPOCH))
    

def load(PATH, model = None, optimizer = None, device = torch.device('cuda')):

    if (model is None):
        model = CRNN().to(device)
    if (optimizer is None):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint = torch.load(PATH)


    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    return model, optimizer, epoch, train_loss, val_loss


def load_bestmodel(PATH, model = None, optimizer = None, device = torch.device('cuda')):

    if (model is None):
        model = CRNN().to(device)
    if (optimizer is None):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    checkpoint = torch.load(os.path.join(PATH, 'bestmodel.pth'))

    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    return model, optimizer, epoch, train_loss, val_loss


# 임시로 추가
def F1_score(mod, TF, t_test):

        TP = ((mod*TF) != 0).sum()
        TN = (( (torch.logical_not(mod.type(torch.bool)))*TF) != 0).sum()
        FP = ((mod*(torch.logical_not(TF.type(torch.bool)))) != 0).sum()
        FN = (((torch.logical_not(mod.type(torch.bool)))*(torch.logical_not(TF.type(torch.bool)))) != 0).sum()
        
        # G_noCC = torch.logical_not(mod[:,0].type(torch.bool))
        # G_noTS = torch.logical_not(mod[:,1].type(torch.bool))
        # T11 = ((mod[:,0]*t_test[:,0]) != 0).sum()
        # T12 = (((mod[:,1]*G_noCC)*t_test[:,0]) != 0).sum()
        # T13 = (((G_noCC*G_noTS)*t_test[:,0]) != 0 ).sum()
        # T21 = (((mod[:,0]*G_noTS)*t_test[:,1]) != 0).sum()
        # T22 = ((mod[:,1]*t_test[:,1]) != 0).sum()
        # T23 = (((G_noCC*G_noTS)*t_test[:,1]) != 0).sum()
        # T = [[T11,T12,T13],[T21,T22,T23]]

        P = torch.true_divide(TP,TP+FP + 0.00001)
        R = torch.true_divide(TP,TP+FN + 0.00001)
        F1 = torch.true_divide(2*P*R , P+R + 0.00001).item()
        # ER = torch.true_divide(FP+FN,TP+FN).item()
        # FPR = torch.true_divide(FP, FP+TN).item()
        # #pdb.set_trace()
        # acc = torch.true_divide(TP+TN , FP+FN+TP+TN).item()
        
        # Total = sum(sum(T, []))
        # RecRate = (T[0][0]+T[1][1]).item()/ Total.item()
        # MissRate = (T[0][2] + T[1][2]).item()/Total.item()
        # ErrorRate = (T[0][1] + T[1][0]).item()/Total.item()

        return F1