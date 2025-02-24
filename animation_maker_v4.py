import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as vutils
import random
import numpy as np
import glob

TOTAL_VERSIONS = 100

#USE ONE OR THE OTHER
model_name = "generator_geo_gen_softnoisy_labels_64"
#model_name = "generator_geo_gen_wasserstein_64"


class Generator(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        return self.main(input)
    


def animate():
    # Set random seed for reproducibility
    manualSeed = 420
    #manualSeed = random.randint(1, 10000) # Uncomment this to get new random results
    print("Random Seed: ", manualSeed)
    pt.manual_seed(manualSeed)
    pt.use_deterministic_algorithms(True)  # Needed for reproducible results

    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    z_size = 100
    z_mean = 0.
    z_std = 1.
    generate_n_img = 16
    columns = 4
    img_list = []
    models_path = 'models/' #Change to the directory where models are
    save_path = 'animations/' #Change to the save directory !!!MUST EXIST BEFORE RUNNING!!!

    z = pt.normal(mean=z_mean, std=z_std, size=(generate_n_img, z_size, 1, 1), device=device)

    for i in range(TOTAL_VERSIONS):
        netGen = pt.load(models_path + model_name + ".py_v{0}.pt".format(i), map_location=device)
        netGen.eval()
        this_images = netGen(z).detach().cpu()
        img_list.append(vutils.make_grid(this_images, nrow=columns, padding=0, normalize=True))
        
    netGen = pt.load(models_path + model_name + ".py_v{0}_final.pt".format(TOTAL_VERSIONS), map_location=device)
    netGen.eval()
    this_images = netGen(z).detach().cpu()
    img_list.append(vutils.make_grid(this_images, nrow=columns, padding=0, normalize=True))

    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    fig.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    ani.save(save_path + model_name + '_animation.gif', writer='ffmpeg', fps=1)
    ani.save(save_path + model_name + '_animation.mp4', writer='ffmpeg', fps=1)
    plt.show()

if __name__ == "__main__":
    animate()