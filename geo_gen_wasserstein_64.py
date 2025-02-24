#!/usr/bin/env python3
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Check if __file__ exists because in colab it doesn't get set
    if "__file__" not in globals():
        file_name = "geo_gen_wasserstein"
    else:
        file_name = os.path.basename(__file__)
    print(file_name)
    
    ### HYPERPARAMETER CONFIGURATION ###
    # Set random seed to ensure reproducibility
    manualSeed = 420
    # manualSeed = random.randint(1, 10000) # Use if you want new results each execution
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    #torch.use_deterministic_algorithms(True) # Force reproducible results
    
    # Root directory of
    dataroot = "Progetto_NN_geogen/dataset"
    
    # Number of workers of dataloader
    workers = 4
    
    # Batch size during training
    batch_size = 1024
    
    # Spatial size of training images. All images will be resized to this size using a transformer
    image_size = 64
    
    # Number of channels of training images (color = 3, B&W = 1)
    nc = 1
    
    # Size of latent vector z (generator input)
    nz = 100

    # Base number of feature maps (and kernels) in generator
    # dimension of output of n-th layer = ngf * 2^(4 - layer_number)
    ngf = 128
    
    # Base number of feature maps (and kernels) in critic
    # ndimension of output of n-th layer = ncf * 2^(layer_number - 1)
    ncf = 128

    # Number of training epochs
    num_epochs = 500
    
    # Learing rate of Adam optimizer
    lr = 0.0002
    
    # Beta1 hyperparameter of Adam optimizer
    beta1 = 0.5
    
    # Number of GPUs available; 0 means CPU mode
    ngpu = 1
    
    
    ### DATASET HANDLING ###
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size), #this is needed because resize can deform the image
                                   transforms.Grayscale(num_output_channels=1), 
                                   transforms.ColorJitter(contrast=(100., 100.)), #since the dataset has many thin grey detail, we change them to black for ease of training
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 0.5),
                               ]))
    
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers)
    
    # Select device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    
    ### WEIGHT INITIALIZATION ###
    # Custom weights initialization for netG and netC
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("LayerNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            #nn.init.constant_(m.bias.data, 0)
            
    
    ### GENERATOR IMPLEMENTATION ###
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            
            self.main = nn.Sequential(
                ## output size for a single channel = (input-1)*stride - 2*padding + kernel_size
                
                #Layer # 1
                #Given input Z, make a Transposed Convolution
                nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
                nn.LayerNorm([ngf*8, 4, 4], bias=False),
                nn.ReLU(inplace=True),
                #output size = ngf*8 x 4 x 4
                
                #Layer # 2
                nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
                nn.LayerNorm([ngf*4, 8, 8], bias=False),
                nn.ReLU(inplace=True),
                #output size = ngf*4 x 8 x 8
                
                #Layer # 3
                nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
                nn.LayerNorm([ngf*2, 16, 16], bias=False),
                nn.ReLU(inplace=True),
                #output size = ngf*2 x 16 x 16
                
                #Layer # 4
                nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                nn.LayerNorm([ngf, 32, 32], bias=False),
                nn.ReLU(inplace=True),
                #output size = ngf x 32 x 32
                
                #Output Layer
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
                #output size = nc x 64 x 64
            )
        def forward(self, input):
            return self.main(input)
    
    # Create a Generator instance    
    netG = Generator().to(device)
    
    # Apply the weights
    netG.apply(weights_init)
    
    # Print the model
    print(netG)
    
    
    ### CRITIC IMPLEMENTATION ###
    class Critic(nn.Module):
        def __init__(self):
            super(Critic, self).__init__()
            
            self.main = nn.Sequential(
                ## output size for a single channel = ⌊(input + 2*padding - kernel_size)/stride⌋ + 1
                
                #Layer # 1
                #Given 64x64 image, make a Convolution
                nn.Conv2d(nc, ncf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #output size = ncf x 32 x 32
                
                #Layer # 2
                nn.Conv2d(ncf, ncf*2, 4, 2, 1, bias=False),
                nn.LayerNorm([ncf*2, 16, 16], bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #output size = ncf*2 x 16 x 16
                
                #Layer # 3
                nn.Conv2d(ncf*2, ncf*4, 4, 2, 1, bias=False),
                nn.LayerNorm([ncf*4, 8, 8], bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #output size = ncf*4 x 8 x 8
                
                #Layer # 4
                nn.Conv2d(ncf*4, ncf*8, 4, 2, 1, bias=False),
                nn.LayerNorm([ncf*8, 4, 4], bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #output size = ncf*8 x 4 x 4
                
                #Output Layer
                nn.Conv2d(ncf*8, 1, 4, 1, 0, bias=False),
                #output size = 1 x 1 x 1
            )
            
        def forward(self, input):
            return self.main(input)
        
    # Create a Critic instance
    netC = Critic().to(device)
    
    # Apply the weights
    netC.apply(weights_init)
    
    # Print the model
    print(netC)
    
    
    ### WASSERSTEIN LOSS FUNCTION IMPLEMENTATION ###
    # Define the gradient penalty function
    def calc_gradient_penalty(critic, real_data, fake_data):
        # Randomly sample from the batch
        alpha = torch.rand(real_data.size(0), 1, 1, 1)
        # Make the random sample the same size as the batch
        alpha = alpha.expand(real_data.size())
        # Move alpha to the same device as the real_data
        alpha = alpha.to(device)
        
        # Make the interpolation between real and fake data
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        # Make interpolates require a gradient
        interpolates.requires_grad_(True)
        # Move interpolates to the same device as alpha and real_data
        interpolates.to(device)
        # Calculate the critic output on the interpolated data
        critic_interpolates = critic(interpolates)
        # Calculate the gradients of the critic output with respect to the interpolated data
        gradients = torch.autograd.grad(outputs=critic_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
                                        create_graph=True,
                                        retain_graph=True,
                                        )[0]
        
        # Flatten the gradients
        gradients = gradients.view(gradients.size(0), -1) # -1 means infer the dimension from the other dimensions
        # Manually calculate norm of gradients to add epsilon
        epsilon = 1e-12
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)
        # Calculate the gradient penalty
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty
    
    # Wasserstein loss definition with gradient penalty
    def wasserstein_loss(real, fake, critic, *, lambda_gp=10):
        # Calculate critic output on real data
        real_crit = critic(real)
        # Calculate critic output on fake data
        fake_crit = critic(fake)
        # Calculate gradient penalty
        gradient_penalty = calc_gradient_penalty(critic, real, fake)
        # Calculate Wasserstein loss
        wasserstein_loss = torch.mean(fake_crit) - torch.mean(real_crit)
        wasserstein_loss_gp = wasserstein_loss + gradient_penalty * lambda_gp
        return wasserstein_loss, wasserstein_loss_gp
    
    
    ### OPTIMIZER CONFIGURATION ###
    # Setup Adam optimizers for G and D
    optimizerC = optim.Adam(netC.parameters(),
                            lr=lr,
                            betas=(beta1, 0.999),
                            weight_decay=2e-5)
    
    optimizerG = optim.Adam(netG.parameters(),
                            lr=lr,
                            betas=(beta1, 0.999),
                            weight_decay=2e-5)
    
    
    ### TRAINING LOOP ###
    
    # Initialize pre-loop variables
    C_losses = []
    G_losses = []
    iters = 0
    version = 0
    N_critic = 5 # Number of times to update critic before updating generator
    save_path = "Progetto_NN_geogen/models/"
    
    print("Starting Training Loop...")
    start = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, start=0):
            
            # (1) Train critic
            for N_iter in range(N_critic):
                # Load batch of real data
                data_real = data[0].to(device)
                # Generate fake data
                noise = torch.randn(size=(data[0].size(0), nz, 1, 1), device=device)
                data_fake = netG(noise)
                # Compute output of critic on real and fake data
                out_real = netC(data_real.detach())
                out_fake = netC(data_fake.detach())
                
                # Compute critic loss
                critic_loss_no_gp, critic_loss = wasserstein_loss(data_real, data_fake, netC)
                # Update critic
                optimizerC.zero_grad()
                critic_loss.backward()
                optimizerC.step()
            
            
            # (2) Train generator
            
            # Generate fake data
            noise = torch.randn(size=(batch_size, nz, 1, 1), device=device)
            data_fake = netG(noise)
            # Compute output of critic on fake data
            out_fake = netC(data_fake)
            
            # Compute generator loss
            generator_loss = -torch.mean(out_fake)
            # Update generator
            optimizerG.zero_grad()
            generator_loss.backward()
            optimizerG.step()
            
            # Output training stats
            end = time.time()
            print('[%d/%d][%d/%d]\tWasserstein: %.4f\tLoss_C: %.4f\tLoss_G: %.4f\tTime: %d'
                    % (epoch + 1,
                        num_epochs, i + 1,
                        len(dataloader),
                        critic_loss_no_gp.item(),
                        critic_loss.item(),
                        generator_loss.item(),
                        end-start)
                    )
            
        # Save Losses for plotting later
        G_losses.append(generator_loss.item())
        C_losses.append(critic_loss.item())
            
        # Save models every 10 epochs
        if ((epoch+1) % 5 == 0):
            torch.save(netG, save_path + "generators/generator_" + file_name + "_v{0}.pt".format(version))
            torch.save(netC, save_path + "critics/critic_" + file_name + "_v{0}.pt".format(version))
            print("Training models version {0} saved in {1}".format(version, save_path))
            version += 1
                
        iters += 1
            
    # Save final version of models
    torch.save(netG, save_path + "generators/generator_" + file_name + "_v{0}_final.pt".format(version))
    print("Final training models (version {0}) saved in {1}".format(version, save_path))
    
    ### PLOT TRAINING GRAPHS ###
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(C_losses, label="C")
    plt.xlabel("Time elapsed")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Progetto_NN_geogen/training_graphs/" + file_name + "_{0}_epochs.png".format(num_epochs))
    #plt.show()
