# ReinforcementLearningImplementations
DDPG and PPO RL agents implemented in Pytorch.   
PPO is synchronous version with only one environment, also PPO does not use random episodes (only last random episode batch, because it is online)
Experiments are tracked using Polyaxon framework and usage is defined in polyaxonfiles (simple YAML definitions)   
CLI usage example:  
bash
```bash
python -u train.py \
                   "--use-monitor" \
                   --device=cpu  \
                   --env="BipedalWalker-v2" \
                   --agent=PPOAgent \
                   --agent-module=agents.PPOAgent.PPOAgent \
                   --random-seed=1024 \
                   --max-episodes=50000 \
                   --max-episode-len=3000 \
                   --random-episodes=0 \
                   --log-dir="./results2" \
                   --lr=0.01 \
                   --gamma=0.99 \
                   --lam=0.9 \
                   --buffer-size=5000000 \
                   --batch-size=8 \
                   --model-dir="./results1/model" \
                   --monitor-dir="./results1/monitor" \
                   --sampler=AdaptiveGaussianSampler \
                   --std=0.6 \
                   --epochs=20
```

cmd
```
python -u train.py ^
                   "--use-monitor" ^
                   --device=cpu  ^
                   --env="LunarLanderContinuous-v2" ^
                   --agent=PPOAgent ^
                   --agent-module=agents.PPOAgent.PPOAgent ^
                   --random-seed=1024 ^
                   --max-episodes=50000 ^
                   --max-episode-len=3000 ^
                   --random-episodes=0 ^
                   --log-dir="./results1" ^
                   --lr=0.01 ^
                   --gamma=0.99 ^
                   --lam=0.9 ^
                   --buffer-size=5000000 ^
                   --batch-size=6 ^
                   --model-dir="./results1/model" ^
                   --monitor-dir="./results1/monitor" ^
                   --sampler=AdaptiveGaussianSampler ^
                   --std=0.6 ^
                   --epochs=20
                   
```













