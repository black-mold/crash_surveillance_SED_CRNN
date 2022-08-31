import argparse
import yaml
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from torch import nn


from utils.dataset import *
from utils.encoder import *
from nnet.CRNN import *
from utils.util import *


# Argument

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--data", 
    default="MIVIA", 
    action="store",
    help="The dataset using training")

parser.add_argument(
    "-c", "--conf_file",
    default="./config/default.yml", 
    help="The configuration file with all the experiment parameters.",
    )

parser.add_argument(
    "-m", "--mode",
    default="train", 
    help="train or test",
    )

parser.add_argument(
    "--gpu",
    default="0",
    type = int,
    help="The number of GPUs to train on, or the gpu to use, default='0', "
    "so uses one GPU",
)

parser.add_argument(
    "--test_from_checkpoint", default=None, help="Test the model specified"
)

args = parser.parse_args()


# yaml load
with open(args.conf_file, "r") as f:
    conf = yaml.safe_load(f)


########################################################################################################################
#                                                     data configuration                                               #
########################################################################################################################

encoder = multi_label_encoder(
    fs = conf['feats']['sample_rate'], 
    audio_len = conf['feats']['audio_len'], 
    n_fft = conf['feats']['n_fft'], 
    hop_length = conf['feats']['hop_length'], 
    net_pooling = conf['feats']['net_pooling']
    )

training_data = mivia(
    conf['data']['audio_folder'], 
    encoder = encoder,
    transform = MelSpectrogram_transform(conf['feats']),
    target_transform= None, 
    partition = conf['data']['select_traindata'], 
    pad_to = conf['feats']['audio_len'],
    fs = conf['feats']['sample_rate']
    )

test_data = mivia(
    conf['data']['audio_folder'], 
    encoder = encoder,
    transform = MelSpectrogram_transform(conf['feats']),
    target_transform= None, 
    partition = conf['data']['select_testdata'], 
    pad_to = conf['feats']['audio_len'],
    fs = conf['feats']['sample_rate']
    )

validation_data = mivia(
    conf['data']['audio_folder'], 
    encoder = encoder,
    transform = MelSpectrogram_transform(conf['feats']),
    target_transform= None, 
    partition = conf['data']['select_valdata'], 
    pad_to = conf['feats']['audio_len'],
    fs = conf['feats']['sample_rate']
    )

nina_test_data = NINA(
    conf['data_crawling']['audio_folder'], 
    transform = MelSpectrogram_transform(conf['feats']),
    target_transform= None, 
    pad_to = conf['data_crawling']['pad_to'],
    fs = conf['data_crawling']['fs']
    )


########################################################################################################################
#                                     device, hyperparameter, tensorboard configuration                                #
########################################################################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# train, validation configuration
learning_rate = conf['opt']['lr'] # 1e-3
batch_size = conf['training']['batch_size']
epochs = conf['training']['n_epochs']
log_dir = conf['training']['log_path']



# log path(tensorboard) make
if args.mode == 'test': # test tensorboard directory configuration
    ckpt_dir = conf['test']['ckpt_path']
    log_dir = os.path.join(log_dir, os.path.basename(ckpt_dir), args.mode) # tensorboard directory
    print(log_dir)
    device = conf['test']['device']
    print(device)

# training / validation tensorboard directory
else: 
    device = conf['training']['device'] 

    ckpt_dir = conf['training']['ckpt_path']
    # save path(model) make
    ckpt_dir = os.path.join(ckpt_dir, args.data + time.strftime('_%Y_%m_%d_%I_%M',time.localtime(time.time()))) # './checkpoint\\MIVIA_2022_07_26_11_26'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
     # tensorboard log directory   
    log_dir = os.path.join(log_dir, args.data + time.strftime('_%Y_%m_%d_%I_%M',time.localtime(time.time())), args.mode) # ./runs\MIVIA_2022_07_26_11_46\train

writer = SummaryWriter(log_dir = log_dir)


train_dataloader = DataLoader(training_data, batch_size, shuffle=True)  # C, D
validation_dataloader = DataLoader(validation_data, 1, shuffle = True)  # B
test_dataloader = DataLoader(test_data, 1, shuffle = False)
nina_test_dataloader = DataLoader(nina_test_data, 1, shuffle = False)             # A



########################################################################################################################
#                                             model, loss function, optimizer                                          #
########################################################################################################################

model = CRNN().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


########################################################################################################################
#                                                         Train                                                        #
########################################################################################################################
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)  # 전체 training data의 size
    num_batches = len(dataloader)   # number of batch

    ## 테스트 더 할라면 여기다가 load 추가, if문이 좋음.

    model.train()

    running_loss = 0.0
    
    for batch_idx, (X, y) in enumerate(dataloader, 0): 
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X[:,:,:3750]) # [batch, frame, class]
        loss = loss_fn(pred, y)  

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss 기록
        running_loss += loss.item()

        print("Train : EPOCH %04d / %04d  | BATCH %04d | LOSS %.4f" %
              (epoch, conf['training']['n_epochs'], batch_idx, running_loss/(batch_idx+1)))

    running_loss = running_loss/num_batches #

    return running_loss



########################################################################################################################
#                                                      VALIDATION                                                      #
########################################################################################################################

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    validation_loss, correct = 0, 0

    score = 0.0 #임시로 추가 1

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X[:,:,:3750])
            validation_loss += loss_fn(pred, y).item()
            correct = 0
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            ### 임시로 추가 2

            label_binary = torch.ge(y, float(conf['test']['threshold']))
            # import pdb
            # pdb.set_trace()

            post = median_filter(pred, conf['test']['median_window'], conf['test']['threshold']) # 이건 계속 쓰는거임. [batchsize, frame, class]
            post = post.to(device)

            is_correct = torch.eq(post, label_binary)
            score += F1_score(post, is_correct,y)

            ### 임시로 추가 

    validation_loss /= num_batches
    correct /= size

    score /= num_batches # 임시로 추가 3

    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validation_loss:>8f}, f1: {score} \n")

    return validation_loss, score


########################################################################################################################
#                                                         TEST                                                         #
########################################################################################################################

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    ### 로드 추가(only model)
    model, _, _, _, _ = load_bestmodel(PATH = conf['test']['ckpt_path'], model = model, optimizer = optimizer, device = torch.device('cuda')) # best model

    model.eval()
    test_loss, correct = 0, 0

    score = 0.0 #임시로 추가 1

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device) #  X   : [batchsize, frequency, frame]
            print(X.shape)
            pred = model(X[:,:,:3750])        # pred : [batchsize, frame, class]
            test_loss += loss_fn(pred, y).item()

            # input feature visualization torch.ge(pred, 0.5)


            ### 임시로 추가 2
            label_binary = torch.ge(y, float(conf['test']['threshold']))

            post = median_filter(pred, conf['test']['median_window'], conf['test']['threshold']) # 이건 계속 쓰는거임. [batchsize, frame, class]
            post = post.to(device)

            is_correct = torch.eq(post, label_binary)
            score += F1_score(post, is_correct,y)
            ### 임시로 추가 

            result_show_graph(spec = X[:,:,:3750], target = y, pred = pred, post = post, i = i, log_dir = log_dir)
            result_show_matrix(spec = X[:,:,:3750], target = y, pred = pred, post = post, i = i, log_dir = log_dir)


            # writer.add_image(f'spectrogram{i}', plot_spectrogram(X.cpu().squeeze(), title = f'{i}'))
            # writer.add_image(f'label{i}', matplotlib_label_show(y.cpu().squeeze(), title = f'label {i}'))
            # writer.add_image(f'prediction{i}', matplotlib_label_show(pred.detach().cpu().squeeze(), title = f'prediction {i}'))
            # writer.add_image(f'post{i}', matplotlib_label_show(mod_binary.detach().cpu().squeeze(), title = f'post {i}'))

            correct = 0
            # correct += (pred.argmax(1) == /y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    score /= num_batches # 임시로 추가 3


    # ...학습 중 손실(running loss)을 기록하고
    writer.add_scalar('test loss', test_loss)
    print(f"test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, f1 : {score} \n")

    return test_loss

def test_crawling(dataloader, model):

    ### 로드 추가(only model)
    model, _, _, _, _ = load_bestmodel(PATH = conf['test']['ckpt_path'], model = model, optimizer = optimizer, device = torch.device('cuda')) # best model

    model.eval()
    # test_loss, correct = 0, 0

    # score = 0.0 #임시로 추가 1

    with torch.no_grad():
        for i, X in enumerate(dataloader):
            # import pdb
            # pdb.set_trace()
            X = X.to(device) #  X   : [batchsize, frequency, frame]
            print(X.shape)
            pred = model(X)        # pred : [batchsize, frame, class]

            # input feature visualization torch.ge(pred, 0.5)


            ### 임시로 추가 2

            post = median_filter(pred, conf['test']['median_window'], conf['test']['threshold']) # 이건 계속 쓰는거임. [batchsize, frame, class]
            post = post.to(device)

            ### 임시로 추가 

            result_show_graph(spec = X, target = None, pred = pred, post = post, i = i, log_dir = log_dir)
            result_show_matrix(spec = X, target = None, pred = pred, post = post, i = i, log_dir = log_dir)          
            

            # writer.add_image(f'spectrogram{i}', plot_spectrogram(X.cpu().squeeze(), title = f'{i}'))
            # writer.add_image(f'label{i}', matplotlib_label_show(y.cpu().squeeze(), title = f'label {i}'))
            # writer.add_image(f'prediction{i}', matplotlib_label_show(pred.detach().cpu().squeeze(), title = f'prediction {i}'))
            # writer.add_image(f'post{i}', matplotlib_label_show(mod_binary.detach().cpu().squeeze(), title = f'post {i}'))

            # correct = 0
            # correct += (pred.argmax(1) == /y).type(torch.float).sum().item()

    # test_loss /= num_batches
    # correct /= size

    # score /= num_batches # 임시로 추가 3


    # # ...학습 중 손실(running loss)을 기록하고
    # writer.add_scalar('test loss', test_loss)
    # print(f"test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, f1 : {score} \n")

    # return test_loss




# Train, Validation the model 

if args.mode == 'train' :

    best_loss = 1000
    best_score = 0.0

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, epoch = t+1)
        validation_loss, score = validation(validation_dataloader, model, loss_fn)

        # tensor board
        writer.add_scalar('training loss', train_loss, t+1)
        writer.add_scalar('validation loss', validation_loss, t+1)

        # model save
        save(EPOCH = t+1, net = model, optimizer = optimizer, TRAIN_LOSS = train_loss, VAL_LOSS = validation_loss, PATH = ckpt_dir)
        if validation_loss < best_loss:

            best_loss = validation_loss
            save_best_model(EPOCH = t+1, net = model, optimizer = optimizer, TRAIN_LOSS = train_loss, VAL_LOSS = validation_loss, f1score = score, PATH = ckpt_dir)
            print(f"--------------------------------------------")

        if score > best_score:

            best_score = score
            save_bestf1_model(EPOCH = t+1, net = model, optimizer = optimizer, TRAIN_LOSS = train_loss, VAL_LOSS = validation_loss, f1score = score, PATH = ckpt_dir)
            print(f"--------------------------------------------")
            
    print("Done!")

# Test the model
else : 
    test_loss = test(test_dataloader, model, loss_fn)
    test_crawling(nina_test_dataloader, model)