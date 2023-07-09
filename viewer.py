import torch
import torch.nn as nn
import torch.optim as optim
from models import *
import random
import argparse
from train import *
from view_atoms_mgmno import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def to_categoricale(a,num_columns):
    s = torch.zeros(num_columns)
    s[a]=1
    return s
f=np.random.randint(0, 8, 10)
#print(f[0])
#print(f)
#print(to_categorical(f,num_columns=8))
#print(to_categoricale(f[0],num_columns=8))
parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=str, default=512+28+1)
parser.add_argument('--latent_dim', type=int, default=512, help='dimensionality of the latent space')
opt = parser.parse_args()
generator = Generator(opt)
discriminator=Discriminator(opt)
net_Q=QHead_(opt)
discriminator.load_state_dict(torch.load('model_cwgan_mgmno/discriminator_100'))
generator.load_state_dict(torch.load('model_cwgan_mgmno/generator_100'))
net_Q.load_state_dict(torch.load("model_cwgan_mgmno/Q_100"))
generator.eval()
discriminator.eval()
net_Q.eval()
def creator(a,b,c):
    fake_c_mg_int=a
    fake_c_mn_int=b
    fake_c_o_int=c
    with torch.no_grad():
        z = autograd.Variable(FloatTensor(np.random.normal(0,1,opt.latent_dim)))
        fake_c_mg=to_categoricale(fake_c_mg_int,num_columns=8)
        fake_c_mn=to_categoricale(fake_c_mn_int,num_columns=8)
        fake_c_o=to_categoricale(fake_c_o_int,num_columns=12)
        natoms_fake =fake_c_mg_int + fake_c_mn_int + fake_c_o_int + 3
        natoms_fake = torch.tensor([natoms_fake/28.0])
#print(z)
#print(fake_c_mg)
#print(fake_c_mn)
#print(fake_c_o)
#print(natoms_fake)
    fake = generator(z,fake_c_mg,fake_c_mn,fake_c_o,natoms_fake)
    print(fake)
    return(fake)
def removezeropadding(fake):
    for i in range(2,30,1):
        for j in range(0,3):
            if (fake[0][0][i][j]<1/6):
                fake[0][0][i][j]=0
            else:
                fake[0][0][i][j]=1.5*(fake[0][0][i][j]-.5)+.5
    return(fake)
def printer(fake,a,b,c):#A,b,c are fake_c_mg_int, fake_c_mn_int, and fake_c_o_int.
    print("\nFractional Coords:")
    for i in range(0,2,1):
        print(fake[0][0][i].detach().numpy())
    print("\nMg Coords (Predicted: "+str(a)+"): ")
    for i in range(2,10,1):
        if ((fake[0][0][i][0]!=0) or (fake[0][0][i][1]!=0) or (fake[0][0][i][2] !=0)):
            print(fake[0][0][i].detach().numpy())
    print("\nMn Coords (Predicted: "+str(b)+"): ")
    for i in range(10,18,1):
        if ((fake[0][0][i][0]!=0) or (fake[0][0][i][1]!=0) or (fake[0][0][i][2] !=0)):
            print(fake[0][0][i].detach().numpy())
    print("\nO Coords (Predicted: "+str(c)+"): ")
    for i in range(18,30,1):
        if ((fake[0][0][i][0]!=0) or (fake[0][0][i][1]!=0) or (fake[0][0][i][2] !=0)):
            print(fake[0][0][i].detach().numpy())
    print("\n\n")
#print(net_Q(fake))
def viewer():
    f = open("train_log_cwgan_mgmno","r")
    y=[]
    for n in f:
        y.append(float(n[(n.index("W")+8):(n.index("W")+16)]))
    x=[r for r in range(0, 504)]
    plt.plot(x,y)
    plt.title("WLoss vs. Generations")
    plt.xlabel("Generations")
    plt.ylabel("WLoss")
    plt.show()
def grapher():
    x=[]
    y=[]
    xremoved=[]
    yremoved=[]
    figure, axis = plt.subplots(2,2)
    area=4
    for i in range(1,300):
        faker=(creator(random.randint(0,7),random.randint(0,7),random.randint(0,11)))
        for i in range(2,30,1):
            x.append(faker[0][0][i][0].item())
            y.append(faker[0][0][i][1].item())
        faker=removezeropadding(faker)
        for i in range(2,30,1):
            xremoved.append(faker[0][0][i][0].item())
            yremoved.append(faker[0][0][i][1].item())
    axis[0,0].scatter(x,y,s=area,alpha=.5)
    axis[0,0].set_title("0 padded")
    axis[0,1].scatter(xremoved,yremoved,s=area,alpha=1)
    axis[0,1].set_title("Non 0 padded")
    plt.show()

def test_images_that_exist():
    gen_images = np.load("Backup7/gen_image_cwgan_mgmno/gen_images_200.npy")
    for xns in gen_images:
        printer(removezeropadding(xns),0,0,0)#Have to get rid of 1 [0] right after fake in printer and removezeropadding

def labeler(fake_c_mg_int,fake_c_mn_int,fake_c_o_int):
    n_mg = fake_c_mg_int+1
    n_mn = fake_c_mn_int+1
    n_o = fake_c_o_int+1
    mg_label_fake_i = torch.tensor(np.array([1]*(n_mg) + [0]*(8-n_mg)))
    mn_label_fake_i = torch.tensor(np.array([1]*(n_mn) + [0]*(8-n_mn)))
    o_label_fake_i = torch.tensor(np.array([1]*(n_o) + [0]*(12-n_o)))
    return mg_label_fake_i, mn_label_fake_i,o_label_fake_i
#train_data = np.load("preparing_dataset/unique_sc_mgmno.npy", allow_pickle=True)
#print(discriminator(train_data))
tester=creator(5,1,6)
#printer(removezeropadding(tester),5,1,6)
#grapher()
#viewer()
view_atoms(tester)
#a,b,c=labeler(5,1,6)


#train_data = np.load("preparing_dataset/mgmno_2000.pickle", allow_pickle=True)
#dataloader = torch.utils.data.DataLoader(train_data, batch_size =32, shuffle = True)
#imgs,label=next(iter(dataloader))
#real_imgs = imgs.view(32, 1, 30,3)
#real_feature,D_real = discriminator(real_imgs)
#print(real_feature)
#print(D_real)
