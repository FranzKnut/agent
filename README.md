# Agent

A Tensorboard plugin to explore reinforcement learning models at the timestep level.

![image](https://user-images.githubusercontent.com/1892071/46906219-61a5b700-cf00-11e8-8e6e-0c821f6ce81f.png)

# Purpose
In 2018 it's surprisingly difficult to understand **why** a reinforcement or inverse reinforcement learning agent makes a decision.

How come the research community lacks usable, open-source interpretability techniques for reinforcement learning like we see from [distill.pub](distill.pub) for supervised learning? Victoria Kraknova made a [well-reasoned call](https://www.youtube.com/watch?v=3HzIutdlpho) for more research in deep RL interpretability for AI Safety at a NIPS workshop a year ago. It seems there is much to be explored about how a RL agent choses actions moment-by-moment and that such work would be valuable, yet the subfield has published little  since 2017. What is causing the paralysis?

The authors of Agent believe a primary bottleneck is misfitted tooling. The current process to extract and save the relevant network activations and episode frames is laborious and complex. Rendering visualizations in a time-dynamic UI is generally intractable in common tools (Jupyter, Tensorboard, Visdom, etc). Even if you succeed, the technique(s) you build tend to be tightly-coupled to your project (see [this group](https://arxiv.org/pdf/1602.02658.pdf) who made an interesting deep RL intepretability tool, but to use it you have to be running Lua and Windows 10). 

The authors of Agent find the above state of affairs frustrating for a [subfield of technical AI Safety](https://medium.com/@deepmindsafetyresearch/building-safe-artificial-intelligence-52f5f75058f1) with low-hanging fruit. RL and IRL research needs a well-documented platform for intepreting agents that is functional across standard workflows. An API you can integrate into your new or existing RL model training code in minutes. A dashboard UI that gives you temporal control and deep visualizations out of the box.

The purpose of Agent is to accelerate progress in deep RL/IRL intepretability to help answer why. For debugging, interest, and research. We are very interested in perspectives from people in the intepretability, deep RL/IRL, and AI Safety communities. Please share your feedback through [GitHub issues](https://github.com/andrewschreiber/agent/issues/new).

# Prototype demo (with sound)

----


# Goals
Agent was built in Python within Tensorboard due to the visualization suite's robustness and popularity among researchers. We hope someday Agent could be merged into Tensorboard itself like the [Beholder plugin](https://github.com/tensorflow/tensorboard/pull/613).

Agent v0 ships end-of-November with two deep learning interpretability techniques, t-SNE and [saliency heatmaps](https://arxiv.org/abs/1711.00138), which we hope will prove immediately useful. Setup should take a few minutes or less. 

For researchers with fresh insight into RL intepretability, Agent v1 will support custom visualizations with the aim to reduce the overhead in developing new techniques by an order of magnitude. 


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
