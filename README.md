# CarRacing-v3 with DQN and DDQN (PyTorch)

This repo solves the CarRacing-v3 environment from the Farama Foundation using DQN and Double DQN (DDQN), implemented in PyTorch. It uses SkipFrame=4 and StackFrame=4 wrappers inspired by the DeepMind paper **"Human-level control through deep reinforcement learning"** by Mnih et al., which addresses learning control policies directly from raw pixels in Atari games. The preprocessing also includes grayscale conversion and resizing frames to (84, 84). Soft updates are used for the target network.



https://github.com/user-attachments/assets/434189dd-823b-44e7-b42d-bd543e00a571



https://github.com/user-attachments/assets/25de9e73-3bb8-470c-8167-a97304273609

