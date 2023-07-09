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
train_data = np.load('preparing_dataset/mgmno_2000.pickle', allow_pickle=True)
dataloader = torch.utils.data.DataLoader(train_data, batch_size =32, shuffle = True)

def creator(a,b,c): #Creates a fake crystal structure with the GAN
    fake_c_mg_int=a
    fake_c_mn_int=b
    fake_c_o_int=c
    d=1
    fake_c_mg_int=torch.tensor([a for n in range(0,d)])
    fake_c_mn_int=torch.tensor([b for n in range(0,d)])
    fake_c_o_int=torch.tensor([c for n in range(0,d)])
    with torch.no_grad():
        z = autograd.Variable(FloatTensor(np.random.normal(0,1,(d,opt.latent_dim))))
        fake_c_mg=to_categorical(fake_c_mg_int,num_columns=8)
        fake_c_mn=to_categorical(fake_c_mn_int,num_columns=8)
        fake_c_o=to_categorical(fake_c_o_int,num_columns=12)
        natoms_fake = fake_c_mg_int + fake_c_mn_int + fake_c_o_int + 3
        natoms_fake=natoms_fake.type('torch.FloatTensor')
        natoms_fake = Variable(natoms_fake/(28.0)).unsqueeze(-1)
    fake = generator(z,fake_c_mg,fake_c_mn,fake_c_o,natoms_fake)
    return(fake.reshape(-1,3))
def viewer(): #Makes the W_Loss vs epoch graph
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
def removezeropaddingprinter(faker):#Does removezeropadding, and also makes it easier to read the output.
    fa=faker[0:2]
    ker=faker[2:30]
    ker,b,c,d=remove_zero_padding(ker)
    final=torch.cat((fa,torch.tensor(ker)),0)
    printed=final.detach().numpy()
    print(printed)
    print("\nFractional Coords:")
    print(printed[0:2])
    print("\nMg Coords:")
    print(printed[2:2+b])
    print("\nMn Coords:")
    print(printed[2+b:2+b+c])
    print("\nO Coords:")
    print(printed[2+b+c:2+b+c+d])
    return(final)
def grapher():#Does the random coord graph
    x=[]
    y=[]
    xremoved=[]
    yremoved=[]
    area=4
    for i in range(1,300):
        faker=(creator(random.randint(0,7),random.randint(0,7),random.randint(0,11)))
        for i in range(2,30,1):
            x.append(faker[i][0].item())
            y.append(faker[i][1].item())
    plt.scatter(x,y,s=area,alpha=.5)
    plt.title("Coordinate Graph")
#    axis[0,1].scatter(xremoved,yremoved,s=area,alpha=1)
#    axis[0,1].set_title("Non 0 padded")
    plt.show()
def W_LossOrigData(numtest):
    s=[]
    for j in range(0,numtest):
        imgs, label = next(iter(dataloader))
        batch_size = imgs.shape[0]
        real_imgs = imgs.view(batch_size, 1, 30,3)
        real_imgs = real_imgs.type('torch.FloatTensor')
        real_feature,D_real=discriminator(real_imgs)
        D_real=D_real.mean()
        fake_feature,D_fake=discriminator(creator(np.random.randint(0,8),np.random.randint(0,8),np.random.randint(0,12)).view(1,1,30,3))#Change to rand
        D_fake=D_fake.mean()
        s.append((D_real-D_fake).detach())
    return(s)

#viewer()
#grapher()
molecule=creator(1,2,4)
view_atoms(molecule) #Displays the molecule made.
