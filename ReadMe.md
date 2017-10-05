# Puzzle RLE
## Project description
Puzzle RLE (Reinforcement Learning Environment) is an environment for learning the puzzle gameplay from the mobile game [Puzzle and Dragons](https://youtu.be/tLku-s20EBE) (Gungho Online Entertainment, Tokyo, Japan).
The environment is a re-implemntation based on orb-matching and clearing mechanics observed during normal gameplay.

The environment supports the following:
* Arbitrary board size and orb types
* pygame environment visualization engine
* 5 Actions: move left, up, right, down, stop-turn
* Optional legal moves
* Combo finding, counting, and clearing with/without skyfall
* Modular reward function design
* Baseline random agent
* Dueling Deep Q-Network agent
* Soon: baseline performance with D-DQN


### Project Milestones & Plans
- Implement basic movement, clearing, skyfall and cascade mechanics ( :heavy_check_mark: Sep 2, '17)
- Implement a random agent and run experiments to generate baseline statistics ( :heavy_check_mark: Sep 2, '17)
- Implement Deep Q Network learning agent (:heavy_check_mark: Sep 9 '17 )
- Validate D-DQN implementation with OpenAI `gym` (:x: target: Sep 18 '17)
- Perform time trials - moves/second, epochs/second (:x:)
- Exceed random baseline performance with DQN agent ( :x: )
- Implement dungeon floor progression with attacking and enemies ( :x: target: Oct '17 )
- Integration with `gym` API/design ( :x: target: Dec '17 )


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
