import minihack_env as me
import matplotlib.pyplot as plt
import matplotlib as mpl
import commons


# How to get a minihack environment from the minihack_env utility.
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id)
state = env.reset()
print("Initial state", state)
next_state = env.step(1)
print("Next State", next_state)




# How to get a minihack environment with also pixels states
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id, add_pixel=True)
state = env.reset()
print("Initial state", state)
plt.imshow(state["pixel"])
plt.show()



# Crop representations to non-empty part
id = me.EMPTY_ROOM
env = me.get_minihack_environment(id, add_pixel=True)
state = env.reset()
print("Initial state", commons.get_crop_chars_from_observation(state))
plt.imshow(commons.get_crop_pixel_from_observation(state))
plt.show()