# ReinforcementLearningImplementations
DDPG and PPO RL agents implemented in Pytorch.   
Experiments are tracked using Polyaxon framework hence usage is defined in polyaxonfiles (simple YAML definitions)   
Usage example:  
bash
```bash
python -u train.py \
       --use-monitor \
       --cuda  \
       --env="Pendulum-v0" \
       --agent=PPOAgent \
       --agent-module=agents.PPOAgent.PPOAgent
       --random-seed=1024 \
       --max-episodes=50000 \
       --max-episode-len=3000 \
       --random-episodes-num=5 \
       --log-dir="./results" \
       --actor-lr=0.01 \
       --critic-lr=0.01 \
       --gamma=0.99 \
       --buffer-size=5000000 \
       --minibatch-size=32 \
       --model-dir="./results/model" \
       --monitor-dir="./results/monitor" \
       --sampler=AdaptiveGaussianSampler
```

cmd
```
python -u train.py ^
                   "--use-monitor" ^
                   "--cuda"  ^
                   --env="Pendulum-v0" ^
                   --agent=PPOAgent ^
                   --agent-module=agents.PPOAgent.PPOAgent ^
                   --random-seed=1024 ^
                   --max-episodes=50000 ^
                   --max-episode-len=3000 ^
                   --random-episodes=0 ^
                   --log-dir="./results" ^
                   --actor-lr=0.01 ^
                   --critic-lr=0.01 ^
                   --gamma=0.99 ^
                   --lam=0.9 ^
                   --buffer-size=5000000 ^
                   --batch-size=32 ^
                   --model-dir="./results/model" ^
                   --monitor-dir="./results/monitor" ^
                   --sampler=GaussianSampler ^
                   --std=0.2
```