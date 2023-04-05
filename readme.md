**Status:** This fork does not support and require tensorflow. Moreover it only supports the pytorch version of ddpg algo with new supported gymnasium instead of OpenAI gym.

Install with (inside root of repo):
```
pip install -e .
```

To start, use the following
```
python3 spinup/algos/pytorch/ddpg/ddpg.py
```

It also reloads the latest model.pt if exists and start human render mode.

One can extract the single actor (and critic) mlp with the following:

```
python3 spinup/algos/pytorch/ddpg/extract_mlp.py
```

To get a destilled student model, one can run the following:

```
python3 spinup/algos/pytorch/ddpg/simple_knowledge_destillation.py
```

All output models are put in `spinup/data/ddpg/ddpg_s0/pyt_save/`

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
