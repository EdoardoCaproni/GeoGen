# Geomorph Generator

Geomorph Generator is an experimental project that applies GAN architectures to the procedural generation of dungeon maps. By leveraging DC-GAN and W-GAN-GP models, this project explores the generation of tessellated "geomorph" tiles, which can be linked together to create maps of adjustable sizes. This approach is particularly useful in game development and tabletop role-playing games (e.g., Dungeons & Dragons), where generating diverse and cost-effective content is key.

 **DC-GAN**
 
![GAN_Geomorph](https://github.com/EdoardoCaproni/GeoGen/blob/main/generator_geo_gen_softnoisy_64.gif)

**W-GAN-GP**

![WGAN_Geomorph](https://github.com/EdoardoCaproni/GeoGen/blob/main/generator_geo_gen_wasserstein_64.gif)

## Overview

- **Objective:** Generate procedural dungeon maps using GANs.
- **Approach:** Instead of creating full maps in one go, generate small, tessellated image tiles ("geomorphs") that can be combined in any orientation. These maps can later be refined via post-processing or manual adjustments.
- **Applications:** Video games, tabletop RPGs, and other environments where dynamic map generation is beneficial.

## Problem Statement

Procedural content generation is critical in the game industry due to its cost efficiency. In tabletop settings, such as battle maps for Dungeons & Dragons, handcrafted maps are time-consuming and expensive. Geomorph Generator addresses this by:

- Using GANs to learn and generate reusable map tiles.
- Allowing the assembly of these tiles into larger, coherent maps.
- Enabling adjustable map sizes through the tessellation of generated geomorphs.

## Dataset

Since there was no pre-made geomorph dataset available, a custom dataset was created with the following steps:

- **Sources:** Assets were collected from public GitHub repositories (e.g., [Dave’s Mapper](https://github.com/davmillar/DavesMapper)) and artists' blogs (e.g., [Aeons & Augauries](https://aeonsnaugauries.blogspot.com/)).
- **Augmentation:** The dataset was enhanced through rotations, flipping, dilation, and erosion.
- **Final Count:** After removing outliers, the dataset comprises 18,912 images.

## Neural Architectures

### DC-GAN

The Deep Convolutional GAN (DC-GAN) replaces fully connected layers with convolutional layers, significantly reducing the number of learnable parameters per layer and enabling a deeper network architecture.

- **Architecture Highlights:**
  - Uses layers such as Conv2D with LeakyReLU activations in the Discriminator.
  - Utilizes ConvTranspose2D with BatchNorm and ReLU in the Generator.
  - The input layer of the Generator and the output layer of the Discriminator have customized stride and padding parameters.
  
- **Training Challenges & Techniques:**
  - The Discriminator can train much faster than the Generator, leading to poor feedback for the Generator.
  - Techniques such as soft labels (real labels in the range 0.7–1.0, fake labels in 0.0–0.3) and a 5% chance of swapping labels were used to mitigate this imbalance.
  - Additional fine-tuning was performed by retraining the Generator with a hand-picked dataset paired with an untrained Discriminator.

### W-GAN-GP

The Wasserstein GAN with Gradient Penalty (W-GAN-GP) replaces the traditional discriminator with a critic that estimates the distance between the generated and real data distributions in Wasserstein space.

- **Architecture Highlights:**
  - The critic provides continuous feedback, which helps the Generator improve steadily.
  - Incorporates Gradient Penalty to enforce the Lipschitz constraint, ensuring the stability of training.
  
- **Benefits:**
  - Provides smoother and more reliable feedback to the Generator.
  - Helps avoid situations where the Generator gets “stuck” due to an overly strong Discriminator.

## Experimental Results

- **Training Losses**
  ![https://github.com/EdoardoCaproni/GeoGen/blob/main/GNN%20Loss.png ](https://github.com/EdoardoCaproni/GeoGen/blob/main/GNN%20Loss.png)
  ![https://github.com/EdoardoCaproni/GeoGen/blob/main/WGNN%20Loss.png](https://github.com/EdoardoCaproni/GeoGen/blob/main/WGNN%20Loss.png)

- **Early Results:**
  - *DC-GAN:* After 320 epochs, experiments at a working resolution of 284×284 (before applying soft & noisy labeling) showed initial promise.
  - *W-GAN:* Early tests at 100 epochs with a working resolution of 64×64 produced more consistent results.
  
- **Final Results:**
  - Due to GPU memory limitations, the W-GAN-GP model was adjusted to produce 64×64 resolution images (which were later resized to 300×300 with contrast adjustments to remove visual artifacts like fog effects).
  - DC-GAN final outputs were similarly post-processed to mitigate issues such as white dots in dungeon walls.

## Acknowledgements
- **Dataset Sources:**
  - *Dave’s Mapper GitHub Repository*
  - *Aeons & Augauries Blog*
- **References:**
  - D. Fernandes e Silva, R. Torchelsen, and M. Aguiar. "Dungeon level generation using generative adversarial network: an experimental study for top-down view games." (2023).
  - I. Gulrajani et al. "Improved Training of Wasserstein GANs." arXiv:1704.00028.
