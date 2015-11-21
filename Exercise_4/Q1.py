from ModNetwork import *

import matplotlib.pyplot as plt
import sys

T  = 1000  # Simulation time

p = 0
mn = ModNetwork(p)

## SIMULATE
for t in xrange(T):  
   sys.stdout.write('Simulating %s ms / %s ms\r' % (t, T - 1))
   sys.stdout.flush()
 
   for i in range(1, EXCIT_MODULES + 1):
     mn.net.layer[i].I = rn.poisson(0.01, 100) * 15
   
   mn.net.Update(t)

# Empty line to prevent overwriting the last simulation line.
print

# Bring all connections into one matrix to display.
# TODO: Ugly
all_connections = []
for fromLayer in range(9):
    for fromNeuron in range(mn.net.layer[fromLayer].N):
      row = []
      for toLayer in range(9):
        for toNeuron in range(mn.net.layer[toLayer].N):
          row.append(mn.net.layer[toLayer].S[fromLayer][toNeuron][fromNeuron])
      
      all_connections.append(row)

# Find the time and neuron ID of all firings.
all_firings_x = []
all_firings_y = []

for layer in range(1, EXCIT_MODULES + 1):
  idx = 100 * (layer - 1)
  
  for i in range(len(mn.net.layer[layer].firings)):
    firing = mn.net.layer[layer].firings[i]
    all_firings_x.append(firing[0])
    all_firings_y.append(idx + firing[1])

np_firings_x = np.array(all_firings_x)
np_firings_y = np.array(all_firings_y)
mean_firings_x = {}
mean_firings_y = {}

for layer in range(EXCIT_MODULES):
  mean_firings_x[layer] = []
  mean_firings_y[layer] = []
  for i in range(0, 49):
    t = 20 * i
    tMin = t - 25
    tMax = t + 25

    inRange = (np_firings_x >= tMin) * (np_firings_x < tMax) * (np_firings_y >= 100 * layer) * (np_firings_y < 100 + 100 * layer)    
    
    idx = np.where(inRange)
    y = len(idx[0]) / 50.0

    mean_firings_y[layer].append(y)
    mean_firings_x[layer].append(t)  

# Plot the connectivity matrix.
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(all_connections)
fig.colorbar(cax)

plt.xlabel('To')
plt.ylabel('From')
plt.show()

plt.figure(1)

# Plot the firings.
plt.subplot(211)
plt.scatter(all_firings_x, all_firings_y)
plt.xlabel('Time (ms) + 0s')
plt.xlim([0, T])
plt.ylabel('Neuron number')
plt.ylim([EXCIT_MODULES * EXCIT_NEURONS_PER_MODULE, 0])
plt.title('p = %s' % p)

# Plot the mean firing rate for each module.
plt.subplot(212)
for layer in mean_firings_x:
  plt.plot(mean_firings_x[layer], mean_firings_y[layer])

plt.xlabel('Time (ms) + 0s')
plt.xlim([0, 1000])
plt.ylabel('Mean firing rate')
plt.show()
