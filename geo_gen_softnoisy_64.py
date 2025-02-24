import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #Check if __file__ exists because in colab it doesn't get set
    if "__file__" not in globals():
      file_name = "geo_gen_more_soft&noisy_labels"
    else:
      file_name = os.path.basename(__file__)

    print(file_name)

    # Set random seed for reproducibility
    manualSeed = 420
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results


    # Root directory for dataset
    dataroot = "Progetto_NN_geogen/dataset"

    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 1024

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 128

    # Size of feature maps in discriminator
    ndf = 128

    # Number of training epochs
    num_epochs = 100

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ColorJitter(contrast=(100., 100.)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 0.5),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # Generator Code

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)


    # Create the generator
    netG = Generator(ngpu).to(device)

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32`
                # `
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)


    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    
    
    #### Training Loop ####

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0
    version = 0
    err_threshold = 0.00009
    err_epochs = 0
    path = "Progetto_NN_geogen/models/"
    
    print("Starting Training Loop...")
    start = time.time()
    
    # Label initialization (whatever value is fine)
    real_label = 1.
    fake_label = 0.
    
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        
        
        
        for i, data in enumerate(dataloader, start = 0):
            
            # Generate soft labels for the current batch
            real_label = np.random.uniform(0.7, 1.0)
            fake_label = np.random.uniform(0.0, 0.3)
            
            # Add a probabily (ex: 5%) that the labels are noisy (flipped)
            if (random.uniform(0.0, 1.0) < 0.05):
                real_label, fake_label = fake_label, real_label
                
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            b = data[0].to(device)
            b_size = b.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(b).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                end = time.time()
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f \t Time: %d'
                      % (epoch + 1, num_epochs, i + 1, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, end-start))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        # Check for anomalous behaviour
        if (errD.item() < err_threshold or
            errG.item() < err_threshold or 
            D_x < err_threshold):
            err_epochs += 1
        else:
            err_epochs = 0
        
        # After 5 epochs of anomalous behaviour, stop training
        if (err_epochs == 5):
            print("Anomalous behaviour detected for 5 epochs! Stopping training.")
            break
        
        # Save models every 10 epochs
        if ((epoch+1) % 5 == 0):
            torch.save(netG, path + "generators/generator_" + file_name + "_v{0}.pt".format(version))
            torch.save(netD, path + "discriminators/discriminator_" + file_name + "_v{0}.pt".format(version))
            print("Training models version {0} saved in {1}".format(version, path))
            version += 1
    
    
    # Save final version of models
    torch.save(netG, path + "generators/generator_" + file_name + "_v{0}_final.pt".format(version))
    torch.save(netD, path + "discriminators/discriminator_" + file_name + "_v{0}_final.pt".format(version))
    print("Final training models (version {0}) saved in {1}".format(version, path))

    # Plot training graphs
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Time elapsed")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Progetto_NN_geogen/training_graphs/" + file_name + "_{0}_epochs.png".format(num_epochs))
    plt.show()