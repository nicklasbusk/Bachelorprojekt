# Bachelor Project

The grade received for our Bachelor thesis was 10/12 on the Danish 7-point scale.

## Algorithmic Pricing Collusion: Introducing Multi-Agent Reinforcement Learning
### Abstract
This thesis explores the phenomenon of algorithmic pricing collusion through the application of Multi-Agent Reinforcement Learning (MARL). The study focuses on a sequential Bertrand duopoly setting, where firms repeatedly adjust their prices in an attempt to maximize profits. The research investigates the collusive behavior of reinforcement learning algorithms; specifically Q learning, WoLF-PHC, and JALAM, under conditions of complete and asymmetric information. Through extensive simulations, it is demonstrated that all three algorithms exhibit collusive tendencies, achieving supra-competitive profits. The study finds that JAL-AM outperforms the other algorithms in profitability, even in the presence of asymmetric information where one firm has incomplete information about the competitor’s prices. These results highlight the robustness of MARL algorithms in adapting to less-than-ideal informational settings. They raise important considerations for policymakers and regulators concerning the potential for tacit collusion in algorithm-driven markets. 

### Code
All code for this project is found in the `code` directory. The code that produces all the relevant figures for the paper can be found in `graphs.ipynb`. The file `model.py` contains the demand, profit, and calculation of epsilon values. The files `q_lib.py`, `WoLF_lib.py`, and `JAL_AM_lib.py` are Python modules that implement Q-learning, WoLF-PHC, and JAL-AM algorithms in sequential Bertrand duopoly. The files `q_learning.ipynb`, `wolf_phc.ipynb`, and `JAL_AM.ipynb` are notebooks used for experimenting with the respective libraries.

To run the Jupyter Notebook (`.ipynb`) files, you can use the Jupyter Notebook application from a distribution like [Anaconda](https://www.anaconda.com/).
