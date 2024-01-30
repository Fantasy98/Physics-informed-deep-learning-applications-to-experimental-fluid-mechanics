"""
Show the summary for all used architectures
"""

from tensorflow.keras import models, layers, optimizers, activations
from time import time

###################
# Burger Eq
###################
print("#"*130)
print("Buerger Eq")
act = activations.tanh
inp = layers.Input(shape = (3,))
hl = inp
for i in range(8):
    hl = layers.Dense(20, activation = act)(hl)
out = layers.Dense(2)(hl)

model = models.Model(inp, out)
print(model.summary())


###################
# Cylinder 
###################
print("#"*130)
print("Vortex Shedding")
nv = 3 #(u, v, p)
act = activations.tanh
inp = layers.Input(shape = (3,))
hl = inp
for i in range(4):
    hl = layers.Dense(20, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())




###################
# minial channel
###################
print("#"*130)
print("Minimal Channel")
nv = 4 #(u, v, w, p)
act = activations.tanh
inp = layers.Input(shape = (4,))
hl = inp
for i in range(10):
    hl = layers.Dense(100, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())


###################
# minial channel
###################
print("#"*130)
print("Hot-wire measurement")
nv = 6 #(u, v, p, uu, vv, uv)
act = activations.tanh
inp = layers.Input(shape = (2,))
hl = inp
for i in range(4):
    hl = layers.Dense(40, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())
