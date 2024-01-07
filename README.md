# D3PM_replication

Implementation (in pytorch) of the D3PM algorithm (diffusion with categorical variable) : https://arxiv.org/pdf/2107.03006.pdf

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

Also we only have 3 bins (categorical variables) in the above exemple images.

Also as the base model we use ContextUnet architecture coming from the deeplearning.ai course on diffusion process : https://deeplearning.ai/short-courses/how-diffusion-models-work/ 

## Notes on the current development

The modification is concern the passage here :

<img width="711" alt="Screenshot 2024-01-07 at 22 24 29" src="https://github.com/Forbu/D3PM_replication/assets/11457947/ae40c3a5-39d3-4da0-9284-dd6a29b93478">

I want to point out that there is currently some (tiny) differences with the paper implementation : 

In our implementation, we currently directly parametrize <img width="96" alt="Screenshot 2024-01-07 at 22 25 21" src="https://github.com/Forbu/D3PM_replication/assets/11457947/069f0427-78e9-46c2-b830-8e324246586a"> as being the neural network and not <img width="78" alt="Screenshot 2024-01-07 at 22 27 11" src="https://github.com/Forbu/D3PM_replication/assets/11457947/472d17fa-acc3-4531-aedd-215f396864e5">


A second interesting point : in the current code we only use the Uniform Noise and not the absorbing noise setup which seems to give better result in the paper.




