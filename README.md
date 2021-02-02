# Implementation of Relational Deep Reinforcement Learning
This Repository is implementation of [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830) with imitation learning on the Warehouse Environment.

The Reinforcement Learning Algorithm is A2C, but it's very easy to change the base algorithm.
## Requirements
- Python: 3.6.1
- Tensorflow: 1.11.0
- Tensorboard:1.11.0
- OpenAI Gym: 0.15.4
- [stable-baselines](https://github.com/hill-a/stable-baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7a1ab7a1c2903e7e1c38756d8cdf7a54a5fd5781e.
    - Already exists in the project, but you need to install the dependencies of stable_baselines


The versions are just what I used and not necessarily strict requirements.

### Install warehouse environment
Go to the `env/warehouse-env/` folder and run the command :
```
python3 setup.py install
```

This will install the warehouse-env environment. Now, you can use this enviroment with the following:
```
from warehouse_env.warehouse_env import WarehouseEnv
import numpy as np
simple_agent = \
         [[ 1,  0,  0,  0,  0,  2, 0],
          [ 0,  0,  0,  0,  0,  0, 0],
          [ 0,  0,  0,  0,  0,  0, 0],
          [ 0,  0,  0,  0,  0,  0, 0],
          [ 0,  0,  0,  0,  0,  0, 0],
          [ 0,  0,  0,  3,  0,  0, 0]]
simple_world = \
         [[  0,  0,  0,  0,  0,  0, 0],
          [  0,  0,  0,  1,  0,  0, 0],
          [  1,  0,  0,  0,  1,  0, 0],
          [  0,  0,  0,  1,  0,  0, 0],
          [  0,  0,  0,  0,  0,  0, 0],
          [  0,  0,  0,  0,  0,  0, 0]]
env = WarehouseEnv(agent_map=np.array(simple_agent), obstacle_map=np.array(simple_world))
```
[More details about the Env](https://github.com/eczy/warehouse-env/blob/pehuen-dev/README.md)

### ODrM* compilation: compile cpp_mstar code
Go to `od_mstar3` folder and run command:
```
python3 setup.py build_ext (may need --inplace as extra argument)
```

Check installation by running:
```
from od_mstar3 import cpp_mstar
```

### Expert Planners 
The following is the example running environment using greedy A* for path planning:
![](gif/gym_animation_astar.gif)

The following is the example running environment using ODrM* for coordinated path planning:
![](gif/gym_animation_mstar.gif)

### Install boxworld environment
Go to the `env/gym-box-world` folder and run the command :
```
pip install -e .
```

## How to Run
All training code is contained within ```main.py```. To view options simply run:
```
python3 main.py --help
```

An example:
```
python3 main.py WarehouseEnv RelationalPolicy -model A2CE -save -total_timesteps 2e7 -env_steps 50 -priming_steps 10000 -coordinated_planner
```

## Experiment result

#### Training Curve

<!-- <div align="center">
<img src="http://ww1.sinaimg.cn/large/74c11ddely1g94sxzhiu2j218g0ukwhe.jpg" width=600 />
</div> -->

<div align="center">
<img src="http://ww1.sinaimg.cn/large/74c11ddely1g9rm3th3b5j20av07n0t0.jpg" width=400 />
</div>


#### Relation diagram
The following is the relation(attention) weight diagram of the agent
> 0:Dark
> 1:White

The greater the weight, the more the color tends to be white
<!-- ![](gif/BoxRandWorldEasy2.gif)
![](gif/BoxRandWorldEasy3.gif) -->

![](gif/concise_cnn_not_reduceObs.gif)
