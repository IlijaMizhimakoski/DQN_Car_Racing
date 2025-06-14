# CarRacing-v3 with DQN and DDQN (PyTorch)

This repo solves the CarRacing-v3 environment from the Farama Foundation using DQN and Double DQN (DDQN), implemented in PyTorch. It uses SkipFrame=4 and StackFrame=4 wrappers inspired by the DeepMind paper **"Human-level control through deep reinforcement learning"** by Mnih et al., which addresses learning control policies directly from raw pixels in Atari games. The preprocessing also includes grayscale conversion and resizing frames to (84, 84). Soft updates are used for the target network.

**DQN model trained for around 600 episodes.** 

![dqngif](https://github.com/user-attachments/assets/f54851a2-f2bc-42e0-9673-903cd6534d43)

**DDQN model trained for around 600 episodes.** 

![ddqngif](https://github.com/user-attachments/assets/bdd66bdb-6b98-4703-86ab-df1445c160c9)

---

*Note: The GIFs above are somewhat choppy due to format limitations. For a smoother and full-quality version, check out the [YouTube video here](https://youtu.be/6AAcua5E8Fw?si=k3cVgLV8tTv_HGt7).*

