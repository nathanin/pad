## Environment notes

### Observations (the State)
Each state has 3 Elements essential to planning:
1. board / orbs on the board
2. position of the selected orb
3. remaining time

In this environment's "text-like" version Element 1 is indicated by an $N \times M$ grid of integers.
Elements 2 and 3 are indicated by a second $N \times M$ grid where the `(n,m)`-th element is a floating point that "ticks" from 1 to 0 with each successive "step" in the episode.

The board state, randomly initialized might look like this:
```
1 2 3 4 3 1
3 4 2 1 2 2
3 3 1 2 4 1
2 1 3 4 3 2
3 3 1 2 4 1
```
Notice we make sure to always begin with a scrambled combination of orbs, without 3+ of the same orb horizontally, or vertically.


A "solved" state might look like this:
```
1 1 1 2 2 2 
2 2 2 3 3 3
4 4 4 2 2 2
3 3 3 1 1 1
2 2 2 4 4 4
```
This state sould evaluate to a reward of 10 combos.

We define an "episode" as a new initialization of orbs.
That is, the identity of orbs cannot change within an episode, and it's possible to find a maximum reward per episode, by the number of 3-pairs that exist in the set.

In an upcoming sprint, the environment will be pulled from a pygame graphics render, in order to emblazen things like selected orb, time, and objectives in pixel-space.

### Action space
Actions are 4-D discrete. 
- left
- right
- up 
- down

At each step, the action is applied to swap the selected orb with the adjacent orb in the direction of the action. 

Actions may be "invalid". 
For instance, on the following board with "`x`" being the selected orb. The valid actions are `(left, up, down)`. The action `right` is invalid because the selected orb has no space off the right of the board.
```
o o o o o o
o o o o o x
o o o o o o
o o o o o o
o o o o o o
```

Invalid actions can either degenerate to `no-op` or in a stricter setting terminate the episode.

### Rewards
The objective is to match 3-or-more "orbs" horizontall or vertically on an $N \times M$ grid.
There are no diagonal matches.
Matches of the same orb colors that touch are merged into a single match.

The combo objective is the number of distinct combos achieved. 

An alternative objective is to essentially sort the orbs into groups of color -- where the reward is number of orbs that are part of any combo ("glob" reward).

A "target" can be set.
In PAD terms the "target" is like leader skill activation: a number of combos, a certain configuration of colors, a number of matches of a certain color, a glob of 5+ of the same orb color, etc.. 
Per-episode, achieving the "target" should come with a high reward.