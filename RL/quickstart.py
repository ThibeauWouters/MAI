import minihack_env as me
import matplotlib.pyplot as plt
import matplotlib as mpl
import commons
import numpy as np


# How to get a minihack environment from the minihack_env utility.
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id)
print(env.__dict__.keys())
state = env.reset()
print("Initial state")
print(state)
next_state, reward, done, some_dict = env.step(1)
print("Next State: ")
print(next_state)
print("Reward: ")
print(reward)
print("Some dict: ")
print(some_dict)

# How to get a minihack environment with also pixels states
# id = me.EMPTY_ROOM
# env = me.get_minihack_environment(id, add_pixel=True)
# state = env.reset()
# print("Initial state", state)
# plt.imshow(state["pixel"])
# plt.savefig("quickstart_with_pixels.png", bbox_inches='tight')
# plt.show()


# Crop representations to non-empty part
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id, add_pixel=True)
state = env.reset()
initial_state = commons.get_crop_chars_from_observation(state)
print("Initial state:")
print(initial_state)
position = np.argwhere(initial_state == 64)[0]
print(position)
plt.imshow(commons.get_crop_pixel_from_observation(state))
plt.savefig("quickstart_with_pixels_crop.png", bbox_inches='tight')
plt.show()

print(env.step(2))
next_state = commons.get_crop_chars_from_observation(state)
print("next_state:")
print(initial_state)
position = np.argwhere(next_state == 64)[0]
print(position)
plt.imshow(commons.get_crop_pixel_from_observation(state))
plt.savefig("quickstart_with_pixels_crop.png", bbox_inches='tight')
plt.show()