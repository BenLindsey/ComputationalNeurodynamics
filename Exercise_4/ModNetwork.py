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
    self.net.layer[0].I = rn.poisson(l, INHIB_NEURONS) * EXTRA_I 

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
      toSize = neurons_per_layer[i]
      net.layer[i].S[0] = rn.uniform(-1, 0, size=(toSize, INHIB_NEURONS))

    # Remove inhib self connections.
    for i in range(INHIB_NEURONS):
      net.layer[0].S[0][i][i] = 0

    # Initialise all excit -> inhib connections to 0
    for excitLayer in range(1, EXCIT_MODULES + 1):
      net.layer[0].S[excitLayer] = np.zeros([INHIB_NEURONS, EXCIT_NEURONS_PER_MODULE])

    # Connect 4 random excit -> inhib
    self._connect_excit_to_inhib(net)

    # Connect excit -> excit in modules
    rewire_set = {}
    for i in range(EXCIT_MODULES):
      (connections, connection_set) = self._create_random_connections(EXCIT_NEURONS_PER_MODULE, CONNECTIONS_PER_MODULE)
      
      # todo pythonically
      for j in range(EXCIT_MODULES):
      	net.layer[i + 1].S[j + 1] = connections if j == i else np.zeros([EXCIT_NEURONS_PER_MODULE, EXCIT_NEURONS_PER_MODULE])

      rewire_set[i + 1] = connection_set

    # Rewire excit -> excit
    self._rewire_net(net, p, rewire_set)

    return net

  def _connect_excit_to_inhib(self, net):
    # Create a random array which contains 25 indexes to each excit module.
    excit_modules = np.arange(1, EXCIT_MODULES + 1)
    duplicates = EXCIT_NEURONS_PER_MODULE / INHIB_INPUTS
    layer_indexes = [excit_modules[i / duplicates] for i in range(EXCIT_MODULES * duplicates)]
    rn.shuffle(layer_indexes)

    # Create an array of unique indexes to each neuron for each excit module.
    module_indexes = [rn.choice(EXCIT_NEURONS_PER_MODULE, \
      EXCIT_NEURONS_PER_MODULE, replace=0).tolist() for i in range(EXCIT_MODULES)]

    # Use layer_indexes to decide which excit layer the inhib neuron should
    # connect to, then use module_indexes to find four excit neurons which
    # haven't been connected to an inhib neuron yet.
    for inhibNeuron in range(INHIB_NEURONS):
      layer = layer_indexes.pop()
      for _ in range(INHIB_INPUTS):
        excitNeuron = module_indexes[layer - 1].pop()

        assert net.layer[0].S[layer][inhibNeuron][excitNeuron] == 0
        net.layer[0].S[layer][inhibNeuron][excitNeuron] = rn.rand()

  def _rewire_net(self, net, p, rewire_set):
    for layer in rewire_set:
      for (start, end) in rewire_set[layer]:
        if (rn.rand() < p):
          assert net.layer[layer].S[layer][end][start] == 1

          net.layer[layer].S[layer][end][start] = 0

          toLayer = 1 + rn.randint(EXCIT_MODULES)
          newEnd = rn.randint(EXCIT_NEURONS_PER_MODULE)
          while net.layer[toLayer].S[layer][newEnd][start] == 1 \
              or (layer == toLayer and newEnd == start):
            toLayer = 1 + rn.randint(EXCIT_MODULES)
            newEnd = rn.randint(EXCIT_NEURONS_PER_MODULE)

          net.layer[toLayer].S[layer][newEnd][start] = 1

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

    for _ in range(connections):
      start = rn.randint(size)
      end = rn.randint(size)
      while connection_matrix[end][start] == 1 or start == end:
        start = rn.randint(size)
        end = rn.randint(size)

      # Add this connection to the matrix and store in set.
      connection_matrix[end][start] = 1
      connection_set.append((start, end))

    return (connection_matrix, connection_set)

