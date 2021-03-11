import ray
from ray.rllib import agents

ray.init() # Skip or set to ignore if already called
config = {'gamma': 0.9,
          'lr': 1e-2,
          # 'num_workers': 4,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}

trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config)
print("train")
for i in range(20):
    results = trainer.train()
    print(results)
print("oo")