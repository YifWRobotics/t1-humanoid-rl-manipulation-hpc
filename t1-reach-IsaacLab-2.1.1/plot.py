import numpy as np
import matplotlib.pyplot as plt

arr = np.load("logs/rsl_rl/t1_flat/7dof-v22-v21-actuated/data/case0/base_pose.npy")

print(arr.shape)  # Check the shape of the array

sample = arr[0]



# 画图
fig, axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
for i in range(7):
    axs[i].plot(sample[:, i], color='C'+str(i))
    axs[i].set_ylabel(f'Feature {i+1}')
    axs[i].grid(True)
    if i == 0:
        axs[i].set_title('Sample 1: Each Feature in Separate Subplot')
axs[-1].set_xlabel('Step')

plt.tight_layout()
plt.show()