import os
import argparse
import time

import torch
import torchvision

from torch import nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchsummary import summary
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-e","--epochs",  type = int, default=50, help="number of epochs in training")
parser.add_argument("-bs","--batch_size",  type = int, default=64, help="batch size for training")
parser.add_argument("-lr","--learning_rate",  type = float , default=1e-3, help="learning rate")
parser.add_argument("-b1","--beta1", type  = float, default = 0.5 , help = "The exponential decay rate for the first moment estimates of adam optimizer")
parser.add_argument("-b2","--beta2", type = float , default = 0.999 , help = "The exponential decay rate for the second moment estimates of adam optimizer")
parser.add_argument("--cuda",action = "store_true",help="Enable CUDA\nNote:If CUDA is not detected model will run on CPU")
parser.add_argument("-q","--quiet", action = "store_true" , help = "Stop printing while training")
parser.add_argument("--interval",type = int ,default=250)
parser.add_argument("--gif",action ="store_true", help = "Create a gif of output images")

args = parser.parse_args()
latent_dim = 100

if args.cuda and torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'


#Loading and Creating Batches of Dataset
transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

mnist = datasets.MNIST(root ='../data/mnist_data',download = True,transform= transform)

data  = torch.utils.data.DataLoader(mnist,batch_size = args.batch_size,shuffle = True,drop_last=True)


#Discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        in_feature = 784
        hid = 256
        out = 1

        self.hidden = nn.Sequential(nn.Linear(in_feature,hid),nn.ReLU(),nn.Dropout(0.1))
        self.out = nn.Sequential(nn.Linear(hid,out),nn.Sigmoid())

    def forward(self,x):
        x = self.hidden(x)
        x = self.out(x)

        return x

#Generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        in_feature = 100
        hid = 256
        out = 784

        self.hidden = nn.Sequential(nn.Linear(in_feature,hid),nn.ReLU(),nn.Dropout(0.1))
        self.out = nn.Sequential(nn.Linear(hid,out),nn.Tanh())

    def forward(self,x):

        x = self.hidden(x)
        x = self.out(x)

        return x


#Creating Model
generator = Generator().to(device)
discriminator = Discriminator().to(device)


#Adam Optimizer
generator_optimizer = optim.Adam(generator.parameters(), lr = args.learning_rate, betas=(args.beta1, args.beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = args.learning_rate, betas=(args.beta1, args.beta2))

# Binary Cross Entropy GAN Loss
Loss = nn.BCELoss()

print('-------------------------------------------------------------------')

print('\tNumber of Epochs     : {}'.format(args.epochs))
print('\tBatch Size           : {}'.format(args.batch_size))
print('\tLearning Rate        : {}'.format(args.learning_rate))
print('\tOptimizer            : {}'.format('Adam'))
print('\tDevice               : {}'.format(device))
print('\tImages Saved after   : {} batches'.format(args.interval))
if args.gif==True:
    print('\tGif Path         : {}'.format('./gan.gif'))

print('-------------------------------------------------------------------')

print('------------------------Generator----------------------------------\n\n')
print("Input Shape {}".format((args.batch_size, 100)))
summary(generator, (args.batch_size, 100))

print('-------------------------------------------------------------------')

print('------------------------Discriminator------------------------------\n\n')
print("Input Shape {}".format((args.batch_size, 784)))
summary(discriminator, (args.batch_size, 784))

print('-------------------------------------------------------------------')


# Creating latent vector of 100 random Variables
def random_input(size):
    r = Variable(torch.randn(size,100)).to(device)
    return r

# Traing Generator
def generator_train(x):

    r = random_input(x.shape[0])

    fake_label_ones = Variable(torch.ones(x.shape[0])).to(device)

    generator_optimizer.zero_grad()

    fake_images = generator(r)

    fake_predict = discriminator(fake_images)

    loss = Loss(fake_predict,fake_label_ones)

    loss.backward()

    generator_optimizer.step()

    return loss

#Training Discriminator
def discriminator_train(x):

    r = random_input(x.shape[0])

    discriminator_optimizer.zero_grad()

    real_label = Variable(torch.ones(x.shape[0])).to(device)

    real_predict = discriminator(x)

    real_loss = Loss(real_predict , real_label)

    real_loss.backward()

    fake_label = Variable(torch.zeros(x.shape[0])).to(device)

    fake_predict = discriminator(generator(r).detach())

    fake_loss = Loss(fake_predict,fake_label)

    fake_loss.backward()

    loss = real_loss + fake_loss

    discriminator_optimizer.step()

    return loss

if 'output' not in os.listdir():
    os.mkdir('output')

total_batch = len(data)

#Storing losses
G_loss = []
D_loss =[]
count = 0

#Training
for epoch in range(args.epochs):
    start = time.time()
    for batch , (image , label) in enumerate(data):

        x = image.view(image.size(0), 784).to(device)

        #Training each network simultaneously 
        d_loss = discriminator_train(x)
        g_loss = generator_train(x)

        D_loss.append(d_loss)
        G_loss.append(g_loss)


        if batch%100==0 and args.quiet ==False:
            print("Epoch ",epoch,'/',args.epochs,'Batch ',batch,'/', total_batch,"Generator Loss:",g_loss.item(),'Discriminator Loss:',d_loss.item())
        
        #Saving Images
        if count % args.interval == 0:
            img = generator(random_input(36)).view(36,1,28,28)
            save_image(img, "output/{}.png".format('0'*(8-len(str(count)))+ str(count)), nrow=6, normalize=True)
    
        count+=1
    print('Epoch {} Done in {} secs !!'.format(epoch+1,time.time()-start))

#Creating GIF
if args.gif == True:
    import imageio
    images = []
    for filename in os.listdir('output'):
        images.append(imageio.imread('output/'+filename))
    imageio.mimsave('gan.gif', images)

#Plotting Loss Curve
plt.plot(G_loss,label='generator')
plt.plot(D_loss,label='discriminator')
plt.legend()
plt.title('Training loss with learning rate: {}'.format(args.learning_rate))
plt.savefig('loss.png')

