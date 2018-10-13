# Puzzle RLE
## Project description
Puzzle RLE (Reinforcement Learning Environment) is an environment for learning the puzzle gameplay from the mobile game [Puzzle and Dragons](https://youtu.be/tLku-s20EBE) (Gungho Online Entertainment, Tokyo, Japan).
The environment is a re-implemntation based on orb-matching and clearing mechanics encountered during normal gameplay.

The environment supports the following:
* pygame environment visualization engine
* 5 Actions: select-orb, move left, up, right, down
* Baseline random agent
* OpenAI Baselines agents

### Project Milestones & Plans
- Implement basic movement, clearing, skyfall and cascade mechanics ( :heavy_check_mark: Sep 2, '17)
- Implement a random agent and run experiments to generate baseline statistics ( :heavy_check_mark: Sep 2, '17)
- Implement Deep Q Network learning agent (:heavy_check_mark: Sep 9 '17 )
- Abandon previous goal of implementing RL algorithms. Use openai baselines, and [stable-baselines](https:github.com/hill-a/stable-baselines) instead! (:heavy_check_mark: Sep '18) 
- Update environment to work with openai gym style with `spaces`, `step()`, ... etc. (:heavy_check_mark: Sep '18)
- Update render-able environment via pygame (:heavy_check_mark: Oct '18)
- Train a successful agent (__in progress__)
- Update agent to take rendered pygame pixels
    - Represent selected orb on-screen
    - Represent environment timer on-screen
    - Move timer to work "real - time" : reset clock when orb is selected. End episode after timer.
    - Allow "unselect" option = "end episode now" <-- estimate that you've gotten the max reward for this episode

## Tested OpenAI baseline agents
|Agent|tested-runing|performance|
| --- | --- | --- |
|DeepQ| no | bad |
|A2C| no | bad |
|HER| no | bad |
|PPO2| yes | bad |

----------------------
During a long hiatus from this project there's been some developments in relational reinforcement learning (arxiv)[https://arxiv.org/pdf/1806.01830]. OpenAI's baselines implementations have been greatly improved and expanded.

#### Depends
* numpy
* tensorflow
* [baselines](https://github.com/openai/baselines) or [stable-baselines](https//:github.com/hill-a/stable-baselines)
* pygame (visualization only)

## References
Environment based on mobile game [Puzzle & Dragons](https://www.puzzleanddragons.us/)

Implementation of DQN algorithms were with reference to the original papers:
* [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/pdf/1312.5602.pdf)
* [Dueling Network Architectures for Deep Reinforcement Learning (2016)](https://arxiv.org/pdf/1511.06581.pdf)

Very much credit to the series of blogposts and Jupyter notebooks by `awjuliani` on reinforement learning:
* [github](https://github.com/awjuliani/DeepRL-Agents)
* [gists](https://gist.github.com/awjuliani?page=1)
* [medium](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

#### License
MIT license ? the free use with citation one.
