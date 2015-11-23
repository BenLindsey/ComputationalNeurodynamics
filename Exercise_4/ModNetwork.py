import numpy as np
import numpy.random as rn

from IzNetwork import IzNetwork

# Network constants from the slides.
EXCIT_MODULES = 8
EXCIT_NEURONS_PER_MODULE = 100

INHIB_NEURONS = 200
INHIB_INPUTS = 4

CONNECTIONS_PER_MODULE = 1000

DMAX = 15

EXTRA_I = 15 # Current to cause spontaneous firing
 
class ModNetwork:

  def __init__(self, p):
    assert 0 <= p <= 1

    self.net = self._build_net(p)

  def update_with_poisson(self, l, t):
    self.net.layer[0].I = np.zeros(INHIB_NEURONS)    

    for i in range(1, EXCIT_MODULES + 1):
      self.net.layer[i].I = rn.poisson(l, EXCIT_NEURONS_PER_MODULE) * EXTRA_I 
    
    self.net.Update(t)

  def _build_net(self, p):
    neurons_per_layer = [INHIB_NEURONS] + [EXCIT_NEURONS_PER_MODULE] * EXCIT_MODULES

    # Create a net where the first layer contains all inhibitory neurons, the
    # remaining layers each contain a module of excitatory neurons.
    net = IzNetwork(neurons_per_layer, DMAX)

    # Turn the first layer into inhibtory neurons.
    self._to_inhibitory_layer(net.layer[0])

    # Turn the remaining layers into excitatory neurons.
    for i in range(EXCIT_MODULES):
      self._to_excitatory_layer(net.layer[i + 1])

    # Set the v, u and firing values for each layer.
    for i in range(len(neurons_per_layer)):
      self._init_layer(net.layer[i])

    # Connect inhib -> everything
    for i in range(len(neurons_per_layer)):
      toSize = INHIB_NEURONS if i == 0 else EXCIT_NEURONS_PER_MODULE
      net.layer[i].S[0] = rn.uniform(-1, 0, size=(toSize, INHIB_NEURONS))

    # Initialise all excit -> inhib connections to 0
    for excitLayer in range(1, EXCIT_MODULES + 1):
      net.layer[0].S[excitLayer] = np.zeros([INHIB_NEURONS, EXCIT_NEURONS_PER_MODULE])

    # Connect 4 random excit -> inhib
    for toInhibNeuron in range(INHIB_NEURONS):
      fromLayer = rn.randint(1, EXCIT_MODULES + 1) 
      fromNeurons = rn.choice(EXCIT_NEURONS_PER_MODULE, INHIB_INPUTS, replace=0)
      for fromNeuron in fromNeurons:
         net.layer[0].S[fromLayer][toInhibNeuron][fromNeuron] = rn.uniform(0, 1)

    # Connect excit -> excit in modules
    rewire_set = {}
    for i in range(EXCIT_MODULES):
      (connections, connection_set) = self._create_random_connections(EXCIT_NEURONS_PER_MODULE, CONNECTIONS_PER_MODULE)
      
      # todo pythonically
      for j in range(EXCIT_MODULES):
      	net.layer[i + 1].S[j + 1] = connections if j == i else np.zeros([EXCIT_NEURONS_PER_MODULE, EXCIT_NEURONS_PER_MODULE])

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

    r = rn.rand(n)

    # Use random values from: http://izhikevich.org/publications/net.m
    layer.a = 0.02 * np.ones(n) + 0.08 * r
    layer.b = 0.25 * np.ones(n) - 0.25 * r
    layer.c = -65 * np.ones(n)
    layer.d = 2 * np.ones(n)
    layer.I = np.zeros(n)

    layer.delay[0] = np.ones([INHIB_NEURONS, INHIB_NEURONS], dtype=int)
    layer.factor[0] = 1

    for i in range(1, EXCIT_MODULES + 1):
      layer.delay[i] = np.ones([INHIB_NEURONS, EXCIT_NEURONS_PER_MODULE])
      layer.factor[i] = 50

  def _to_excitatory_layer(self, layer):
    n = layer.N

    r = rn.rand(n)
 
    layer.a = 0.02 * np.ones(n)
    layer.b = 0.20 * np.ones(n)
    layer.c = -65 + 15*(r**2)
    layer.d = 8 - 6*(r**2) 

    layer.delay[0] = np.ones([EXCIT_NEURONS_PER_MODULE, INHIB_NEURONS])
    layer.factor[0] = 2 

    for i in range(1, EXCIT_MODULES + 1):
      layer.delay[i] = rn.randint(low=1, high=20, size=(EXCIT_NEURONS_PER_MODULE, EXCIT_NEURONS_PER_MODULE)) 
      layer.factor[i] = 17 

  def _init_layer(self, layer):
    layer.v = -65 * np.ones(layer.N)
    layer.u = layer.b * layer.v
    layer.firings = np.array([])

  def _create_random_connections(self, size, connections):
    connection_matrix = np.zeros([size, size])
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

