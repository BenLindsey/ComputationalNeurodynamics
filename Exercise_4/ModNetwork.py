import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from IzNetwork import IzNetwork

# Network constants from the slides.
EXCIT_MODULES = 8
EXCIT_NEURONS_PER_MODULE = 100

INHIB_NEURONS = 200
INHIB_INPUTS = 4

CONNECTIONS_PER_MODULE = 1000

T  = 500  # Simulation time
Ib = 5    # Base current
 
class ModNetwork:

  def __init__(self, p):
    assert 0 <= p <= 1

    self.net = self._build_net(p)

  def _build_net(self, p):
    dmax = 1
    neurons_per_layer = [INHIB_NEURONS] + [EXCIT_NEURONS_PER_MODULE] * EXCIT_MODULES

    # Create a net where the first layer contains all inhibitory neurons, the
    # remaining layers each contain a module of excitatory neurons.
    net = IzNetwork(neurons_per_layer, dmax)

    # Turn the first layer into inhibtory neurons.
    self._to_inhibitory_layer(net.layer[0])

    # Turn the remaining layers into excitatory neurons.
    for i in range(EXCIT_MODULES):
      self._to_excitatory_layer(net.layer[i + 1])

    for i in range(len(neurons_per_layer)):
      self._init_layer(net.layer[i])
 
   # map(self._init_layer, net.layer)

    # Connect inhib -> everything
    for i in range(len(neurons_per_layer)):
      net.layer[i].S[0] = np.ones([INHIB_NEURONS if i == 0 else EXCIT_NEURONS_PER_MODULE, INHIB_NEURONS]) 

    # Initialise all excit -> inhib connections to 0
    for excitLayer in range(1, EXCIT_MODULES + 1):
      net.layer[0].S[excitLayer] = np.zeros([INHIB_NEURONS, EXCIT_NEURONS_PER_MODULE])

    # Connect 4 random excit -> inhib
    for toInhibNeuron in range(INHIB_NEURONS):
      fromLayer = rn.randint(1, EXCIT_MODULES + 1) 
      fromNeurons = rn.choice(EXCIT_NEURONS_PER_MODULE, INHIB_INPUTS, replace=0)
      for fromNeuron in fromNeurons:
         net.layer[0].S[fromLayer][toInhibNeuron][fromNeuron] = 1  

    # Connect excit -> excit in modules
    rewire_set = {}
    for i in range(EXCIT_MODULES):
      (connections, connection_set) = self._create_random_connections(EXCIT_NEURONS_PER_MODULE, CONNECTIONS_PER_MODULE)
      
      # todo pythonically
      for j in range(EXCIT_MODULES):
      	net.layer[i + 1].S[j + 1] = connections if j == i else [[0 for x in range(EXCIT_NEURONS_PER_MODULE)] for y in range(EXCIT_NEURONS_PER_MODULE)]

      rewire_set[i + 1] = connection_set

    # Rewure excit -> excit
    return self._rewire_net(net, p, rewire_set)

  def _rewire_net(self, net, p, rewire_set):
    for layer in rewire_set:
      for (start, end) in rewire_set[layer]:
        if (rn.rand() < p):
          assert net.layer[layer].S[layer][end][start] == 1

          net.layer[layer].S[layer][end][start] = 0

          # todo Must go to another layer? Can go to inhib layer?
          toLayer = rn.randint(net.Nlayers)
          newEnd = rn.randint(net.layer[toLayer].N)

          net.layer[toLayer].S[layer][newEnd][start] = 1
    return net

  def _to_inhibitory_layer(self, layer):
    n = layer.N

    layer.a = 0.02 * np.ones(n)
    layer.b = 0.25 * np.ones(n)
    layer.c = -65 * np.ones(n)
    layer.d = 2 * np.ones(n)
    layer.I = np.zeros(n)

# net.layer[1].factor[0] 
    layer.delay[0] = np.ones([INHIB_NEURONS, INHIB_NEURONS], dtype=int)
    layer.factor[0] = 1

    for i in range(1, EXCIT_MODULES + 1):
      layer.delay[i] = np.ones([INHIB_NEURONS, EXCIT_NEURONS_PER_MODULE])
      layer.factor[i] = 2

  def _to_excitatory_layer(self, layer):
    n = layer.N

    layer.a = 0.02 * np.ones(n)
    layer.b = 0.2 * np.ones(n)
    layer.c = -65 * np.ones(n)
    layer.d = 8 * np.ones(n) 
    layer.I = Ib * np.ones(n)

    layer.delay[0] = np.ones([EXCIT_NEURONS_PER_MODULE, INHIB_NEURONS])
    layer.factor[0] = 50 

    for i in range(1, EXCIT_MODULES + 1):
      layer.delay[i] = rn.randint(low=1, high=20, size=(EXCIT_NEURONS_PER_MODULE, EXCIT_NEURONS_PER_MODULE)) 
      layer.factor[i] = 17 

  def _init_layer(self, layer):
    layer.v = -65 * np.ones(layer.N)
    layer.u = layer.b * layer.v
    layer.firings = np.array([])

  def _create_random_connections(self, size, connections):
    connection_matrix = [[0 for x in range(size)] for y in range(size)]
    connection_set = []

    for i in range(connections):
      start = rn.randint(size)
      end = rn.randint(size)
      while connection_matrix[end][start] == 1:
        start = rn.randint(size)
        end = rn.randint(size)

      # Add this connection to the matrix and store in set.
      connection_matrix[end][start] = 1
      connection_set.append((start, end))

    return (connection_matrix, connection_set)

mn = ModNetwork(0.1)

## SIMULATE
#for t in xrange(T):
#   mn.net.Update(t)

# Bring all connections into one matrix to display.
# TODO: Ugly
all_connections = []
all_firings = []
for fromLayer in range(9):
  for fromNeuron in range(mn.net.layer[fromLayer].N):
    row = []
    for toLayer in range(9):
      for toNeuron in range(mn.net.layer[toLayer].N):
        row.append(mn.net.layer[toLayer].S[fromLayer][toNeuron][fromNeuron])
      
    all_connections.append(row)

plt.matshow(all_connections)
plt.xlabel('To')
plt.ylabel('From')
plt.show()

