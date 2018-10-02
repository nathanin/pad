# Puzzle RLE
## Project description
Puzzle RLE (Reinforcement Learning Environment) is an environment for learning the puzzle gameplay from the mobile game [Puzzle and Dragons](https://youtu.be/tLku-s20EBE) (Gungho Online Entertainment, Tokyo, Japan).
The environment is a re-implemntation based on orb-matching and clearing mechanics observed during normal gameplay.

The environment supports the following:
* pygame environment visualization engine
* 5 Actions: select-orb, move left, up, right, down, stop-turn
* Baseline random agent
* Dueling Deep Q-Network agent

### Project Milestones & Plans
- Implement basic movement, clearing, skyfall and cascade mechanics ( :heavy_check_mark: Sep 2, '17)
- Implement a random agent and run experiments to generate baseline statistics ( :heavy_check_mark: Sep 2, '17)
- Implement Deep Q Network learning agent (:heavy_check_mark: Sep 9 '17 )

----------------------
During a long hiatus from this project there's been some developments in relational reinforcement learning (arxiv)[https://arxiv.org/pdf/1806.01830]. In addition, there's been the release and stabilization of TensorFlow Eager execution.
The new plan is to implement an agent with a relational reinforcement module, and to translate the whole project to the tf.keras.Model pattern.

I'm going to edit the main objective. Reward will be calculated as +10 for N combos, at which point the board terminates. -1 Reward will be given if the agent fails to reach N combos with M moves. +1 will be given for each additional combo, not counting broken combos. In other words, switching two orbs back and forth to make/break a combo will give +1 for the first match, and +0 the rest.

#### Depends
* numpy
* tensorflow
* pygame (visualization only)

## References
Implementation of DQN algorithms were with reference to the original papers:
* [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/pdf/1312.5602.pdf)
* [Dueling Network Architectures for Deep Reinforcement Learning (2016)](https://arxiv.org/pdf/1511.06581.pdf)

Very much credit to the series of blogposts and Jupyter notebooks provided by `awjuliani` on reinforement learning:
* [github](https://github.com/awjuliani/DeepRL-Agents)
* [gists](https://gist.github.com/awjuliani?page=1)
* [medium](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

#### License
MIT license ? the free use with citation one.
