# D3PM_replication

Implementation of the D3PM algorithm (diffusion with categorical variable) : https://arxiv.org/pdf/2107.03006.pdf

In this repository we will implement the D3PM algorithm on discrete MNIST data. 

## Forward discrete diffusion process 

in the forward discrete we have something like that :

![image](https://github.com/Forbu/D3PM_replication/assets/11457947/c932140e-25a9-4eca-a900-cca43e915aa0)


## Reverse diffusion process after training

Training is currently being done.

Result after training on 10% of the data points :

![image](https://github.com/Forbu/D3PM_replication/assets/11457947/abd70b55-6830-47f3-9f76-28403e002af6)

## Training params

We train using nb_temporal_step = 254, lambda = 0.01 and hidden_dim = 32 (no need to have huge number of latent images).

Also as the base model we use ContextUnet architecture coming from the deeplearning.ai course on diffusion process : https://deeplearning.ai/short-courses/how-diffusion-models-work/ 
