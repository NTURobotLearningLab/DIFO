import matplotlib.pyplot as plt
import numpy as np


def min_max_norm(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))


np.random.seed(0)
# sine
f = 3
x = np.linspace(0, 1, 500, endpoint=True)
y = min_max_norm(np.sin(2 * np.pi * f * x) + x * 4)
# collect data
scale = 20
s = np.repeat(x, scale)
s_t_plus_1 = np.repeat(y, scale)
noise = np.random.normal(0, 0.05, s_t_plus_1.shape)
s_t_plus_1_noise = s_t_plus_1 + noise
s_s_t_plus_1_noise = np.append(s, min_max_norm(s_t_plus_1_noise))
s = s_s_t_plus_1_noise[: len(s)]
s_t_plus_1_noise = s_s_t_plus_1_noise[len(s) :]
a = s_t_plus_1_noise - s

print(max(s), min(s))
print(max(a), min(a))
plt.scatter(s, s_t_plus_1_noise, s=3, alpha=0.02)
plt.xlabel("state")
plt.ylabel("next state")
plt.title(f"a = ({max(a):.4f},{min(a):.4f})")
plt.suptitle(f"Sine Wave")
plt.savefig("sine_wave.png")
