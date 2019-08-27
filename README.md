# ReinforcementLearningImplementations
DDPG and PPO RL agents implemented in Pytorch.   
Experiments are tracked using Polyaxon framework hence usage is defined in polyaxonfiles (simple YAML definitions)   
Usage example:  
bash
```bash
python -u train.py \
                   "--use-monitor" \
                   --env="Pendulum-v0" \
                   --agent=PPOAgent \
                   --agent-module=agents.PPOAgent.PPOAgent \
                   --random-seed=1024 \
                   --max-episodes=50000 \
                   --max-episode-len=3000 \
                   --random-episodes=0 \
                   --log-dir="./results1" \
                   --actor-lr=0.001 \
                   --critic-lr=0.004 \
                   --gamma=0.9 \
                   --lam=0.9 \
                   --buffer-size=5000000 \
                   --batch-size=4 \
                   --model-dir="./results1/model" \
                   --monitor-dir="./results1/monitor" \
                   --sampler=GaussianSampler \
                   --std=0.6 \
                   --epochs=1 \
                   --checkpoint-steps=100
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