# Agent

A Tensorboard plugin to explore reinforcement learning models at the timestep level.

![image](https://user-images.githubusercontent.com/1892071/46906219-61a5b700-cf00-11e8-8e6e-0c821f6ce81f.png)

# Purpose
In 2018 it's surprisingly difficult to understand why a RL or IRL agent makes a decision.

Why don't we have interpretability techniques in reinforcement learning like we see from distill.pub in supervised learning? Victoria Kraknova made a [well-reasoned call](https://www.youtube.com/watch?v=3HzIutdlpho) for research in deep RL interpretability for AI Safety at a NIPS workshop a year ago and yet the field has published little progress since. 

The authors of Agent believe the bottleneck is misfitted tooling. The current process to extract and save the relevant network activations and episode frames is laborious. Rendering visualizations in a time-dynamic UI is generally intractable. It's hard to get setup, then the visualization you built tends to be tightly-coupled to your project (see this group who made an interesting deep RL intepretability tool, but it requires Lua and Windows 10). 

The authors of Agent find the above state of affairs unacceptable for a subfield of technical AI Safety with substantial low-hanging fruit. RL and IRL needs a well-documented platform for intepreting agents functional across standard workflows. Agent v0 will ship with a two deep learning interpretability techniques out of the box, t-SNE and saliency heatmaps, which we hope will prove immediately useful for debugging. For researchers with fresh insight into RL intepretability, Agent v1 will support custom visualizations through a python subclass with the aim to reduce the overhead in developing new techniques by an order of magnitude.

# Prototype demo (with sound)
----

## Setup (Work in progress)
### Note: Agent is currently built for demonstration purposes.
Packages required (recommended version):

  Python virtual environment (3.6)

  [Bazel](https://docs.bazel.build/versions/master/install.html) build tool from Google. Install guide in link. (0.17.2)

  Tensorflow (1.11)
  
Then:

    git clone https://github.com/andrewschreiber/agent.git
    cd agent
    
    # Install API layer in your Python virtual environment
    pip install .

    #Build takes ~7m on a 2015 Macbook
    bazel build tensorboard:tensorboard
    
    #Use the custom tensorboard build by running
    ./bazel-bin/tensorboard/tensorboard --logdir tb/logdirectory/logs
    

### Tensorboard
To visualize training, use the following command to setup Baselines to
send tensorboard log files.

    export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=logs

Return to the original terminal tab, at the root of rlmonitor, and run your training:

    python -m baselines.run --alg=deepq --env=CartPole-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5

Go to the linked URL in the tensorboard tab to see your model train.

### Run Cartpole with DQN
    cd examples/baselines

Follow instuctions from https://github.com/andrewschreiber/baselines to
install Gym. Then:

Train a model:

    python -m baselines.run --alg=deepq --env=CartPole-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5

See the model playing Carptole:

    python -m baselines.run --alg=deepq --env=CartPole-v0 --load_path=./cartpole_model.pkl --num_timesteps=0 --play
