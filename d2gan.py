#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle
import sys

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),        # 32→16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),    # 16→8
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),  # 8→4
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),      # 4→1
            nn.Softplus()
        )

    def forward(self, input):
        return self.main(input)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DCGAN")

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs to train',
        required=True
    )

    parser.add_argument(
        '--checkpoint',
        type=int,
        help='Checkpoint epoch to resume from'
    )

    args = parser.parse_args()
        
    num_epochs=args.epochs
    print(f"Number of epochs: {num_epochs}")
    checkpoint_epoch=None
    try:
        checkpoint_epoch=args.checkpoint
        print(f"checkpoint epoch: {checkpoint_epoch}")
    except:
        pass

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 64

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 32
    num_eval_images = 10000  # Total images needed for a reliable score

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    alpha=0.2
    beta=0.2

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu).to(device)
    netD1 = Discriminator(ngpu).to(device)
    netD2 = Discriminator(ngpu).to(device)
    optimizerD1 = optim.Adam(netD1.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD2 = optim.Adam(netD2.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    dataloader_generator = torch.Generator()
    G_losses, D1_losses,D2_losses, img_list = [], [], [], []
    iters, start_epoch = 0, 0
    fixed_eval_noise=None
    fixed_visualisation_noise=None
    if checkpoint_epoch:
        print("-----------Using saved model---------")
        checkpoint_path = f"checkpoint_epoch_{checkpoint_epoch}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD1.load_state_dict(checkpoint['netD1_state_dict'])
        netD2.load_state_dict(checkpoint['netD2_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD1.load_state_dict(checkpoint['optimizerD1_state_dict'])
        optimizerD2.load_state_dict(checkpoint['optimizerD2_state_dict'])
        G_losses = checkpoint['G_losses']
        D1_losses = checkpoint['D1_losses']
        D2_losses = checkpoint['D2_losses']
        img_list = checkpoint['img_list']
        iters = checkpoint['iters']
        start_epoch = checkpoint['epoch'] + 1
        # Restore RNG state IMMEDIATELY
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
        dataloader_generator.set_state(checkpoint['dataloader_rng_state'].cpu())

        
        fixed_visualisation_noise = checkpoint['fixed_visualisation_noise'] # for plotting  
        fixed_eval_noise=checkpoint['fixed_eval_noise'] # for inception score calculation
    if not checkpoint_epoch:
        fixed_visualisation_noise = torch.randn(64, nz, 1, 1, device=device)
        fixed_eval_noise=torch.randn(num_eval_images, nz, 1, 1, device=device)
        # Apply weights_init only when training from scratch
        netG.apply(weights_init)
        netD1.apply(weights_init)
        netD2.apply(weights_init)

        
        dataloader_generator.manual_seed(manualSeed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1,1]
    ])

    dataset = dset.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers,generator=dataloader_generator)

    '''
    
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    '''
    

    criterion = nn.BCELoss()


    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    epoch_cut_off=0


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch,num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ##############################################
            # (1) Update D1 and D2 networks
            ##############################################
            netD1.zero_grad()
            netD2.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise).detach()

            # D1 loss: maximize α * log(D1(x)) + D1(G(z)) -> minimize negative
            D1_real = netD1(real_cpu).view(-1)
            D1_fake = netD1(fake).view(-1)
            d1_loss = torch.mean(-alpha * torch.log(D1_real + 1e-8) + D1_fake)

            # D2 loss: maximize D2(x) + β * log(D2(G(z))) -> minimize negative
            D2_real = netD2(real_cpu).view(-1)
            D2_fake = netD2(fake).view(-1)
            d2_loss = torch.mean(D2_real - beta * torch.log(D2_fake + 1e-8))

            d_loss = d1_loss + d2_loss
            d_loss.backward()
            optimizerD1.step()
            optimizerD2.step()

            ##############################################
            # (2) Update Generator
            ##############################################
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            D1_fake_forG = netD1(fake).view(-1)
            D2_fake_forG = netD2(fake).view(-1)

            g_loss = torch.mean(-D1_fake_forG + beta * torch.log(D2_fake_forG + 1e-8))
            g_loss.backward()
            optimizerG.step()


            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D1: %.4f\tLoss_D2: %.4f\tLoss_G: %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        d1_loss.item(),d2_loss.item(), g_loss.item()))

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D1_losses.append(d1_loss.item())
            D2_losses.append(d2_loss.item())
            
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_visualisation_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            
            iters += 1
        
        if((epoch+1)%5==0 and (epoch+1)>=epoch_cut_off):
            checkpoint = {
            'epoch': epoch,
            'iters': iters,
            'netG_state_dict': netG.state_dict(),
            'netD1_state_dict': netD1.state_dict(),
            'netD2_state_dict': netD2.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD1_state_dict': optimizerD1.state_dict(),
            'optimizerD2_state_dict': optimizerD2.state_dict(),
            'G_losses': G_losses,
            'D1_losses': D1_losses,
			'D2_losses': D2_losses,
            'img_list': img_list,
            'fixed_visualisation_noise': fixed_visualisation_noise,
            'fixed_eval_noise':fixed_eval_noise,
            'rng_state': torch.get_rng_state(), # Save CPU RNG state
            'cuda_rng_state': torch.cuda.get_rng_state(), # Save current GPU RNG state
            'dataloader_rng_state': dataloader_generator.get_state()
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')


            print(f"--- Generating a fixed set of images for evaluation at epoch {epoch+1} ---")            
            # List to hold the generated batches
            eval_images = []
            eval_batch_size = 64
            
            # Set model to evaluation mode
            netG.eval()
            
            with torch.no_grad():
                for i in range(0, num_eval_images, eval_batch_size):
                    noise_batch = fixed_eval_noise[i:i+eval_batch_size]
                    fake_batch = netG(noise_batch.to(device)).detach().cpu()
                    eval_images.append(fake_batch)
                    

            # Combine all the generated batches into a single tensor
            all_fake_images_tensor = torch.cat(eval_images, dim=0)
            torch.save(all_fake_images_tensor, f"all_fake_images_epoch_{epoch+1}.pt")



            
            # Set model back to training mode
            netG.train()
                
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D1_losses,label="D1")
    plt.plot(D2_losses,label="D2")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i.cpu(),(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.savefig("fake_images_last_epoch.png", bbox_inches='tight')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("fake_images_last_epoch.png", bbox_inches='tight')
    plt.show()
    


