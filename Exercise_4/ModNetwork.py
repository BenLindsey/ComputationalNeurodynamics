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

    # Create the connections in each module.
    rewire_set = {}
    for i in range(EXCIT_MODULES):
      (connections, connection_set) = self._create_random_connections(EXCIT_NEURONS_PER_MODULE, CONNECTIONS_PER_MODULE)

      net.layer[i + 1].S[i + 1] = connections
      rewire_set[i + 1] = connection_set

    return self._rewire_net(net, p, rewire_set)

  def _rewire_net(self, net, p, rewire_set):
    for layer in rewire_set:
      for (start, end) in rewire_set[layer]:
        if (rn.rand() < p):
          assert net.layer[layer].S[layer][end][start] == 1

          net.layer[layer].S[layer][end][start] = 0

          # Must go to another layer?
          toLayer = rand(net.Nlayers)
          end = rand(net.layer[toLayer].N)

    return net

  def _to_inhibitory_layer(self, layer):
    n = layer.N

    layer.a = 0.02 * np.ones(n)
    layer.b = 0.25 * np.ones(n)
    layer.c = -65 * np.ones(n)
    layer.d = 2 * np.ones(n)

  def _to_excitatory_layer(self, layer):
    n = layer.N

    layer.a = 0.02 * np.ones(n)
    layer.b = 0.2 * np.ones(n)
    layer.c = -65 * np.ones(n)
    layer.d = 8 * np.ones(n)

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

# Bring all connections into one matrix to display.
# TODO: Ugly
all_connections = []
for y in range(9):
  for i in range(mn.net.layer[y].N):
    row = []
    for x in range(9):
      if x in mn.net.layer[y].S:
        row.extend(mn.net.layer[y].S[x][i])
      else:
        row.extend([0] * mn.net.layer[x].N)
    all_connections.append(row)

plt.matshow(all_connections)
plt.show()

