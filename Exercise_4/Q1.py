from ModNetwork import *

import matplotlib.pyplot as plt

# TODO: Check that these dependencies are allowed.
import sys
import os

T  = 1000  # Simulation time
PLOT_OUTPUT_DIR = 'plots/'

def main():
  # Ensure that the directory for the plots exists.
  if not os.path.exists(PLOT_OUTPUT_DIR):
      os.makedirs(PLOT_OUTPUT_DIR)

  p_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

  for p in p_values:
    print
    print 'Network with p = %s:' % p

    mn = ModNetwork(p)
    
    # Bring all connections into one matrix to display.
    all_connections = get_connection_matrix(mn.net)
    
    # Plot the connectivity matrix.
    plot_connectivity_matrix(p, all_connections)
    
    # Simulate
    run_net(mn)
    
    # Find the time and neuron ID of all firings.
    (all_firings_x, all_firings_y) = get_firings(mn.net)
    
    # Find the mean firings for each module
    np_firings_x = np.array(all_firings_x)
    np_firings_y = np.array(all_firings_y)
    (mean_firings_x, mean_firings_y) = get_mean_firings(np_firings_x, np_firings_y)

    # Plot firings and mean firings.
    plot_firings(p, all_firings_x, all_firings_y, mean_firings_x, mean_firings_y)

def run_net(mn):
  for t in xrange(T):  
     sys.stdout.write('Simulating %s ms / %s ms\r' % (t, T - 1))
     sys.stdout.flush()
   
     mn.update_with_poisson(0.01, t)
  
  # Empty line to prevent overwriting the last simulation line.
  print

def get_connection_matrix(net):
  all_connections = []
  for fromLayer in range(9):
    for fromNeuron in range(net.layer[fromLayer].N):
      row = []
      for toLayer in range(9):
        for toNeuron in range(net.layer[toLayer].N):
          row.append(net.layer[toLayer].S[fromLayer][toNeuron][fromNeuron])
      
      all_connections.append(row)
  return all_connections

def get_firings(net):
  all_firings_x = []
  all_firings_y = []
  
  for layer in range(1, EXCIT_MODULES + 1):
    idx = 100 * (layer - 1)
    
    for i in range(len(net.layer[layer].firings)):
      firing = net.layer[layer].firings[i]
      all_firings_x.append(firing[0])
      all_firings_y.append(idx + firing[1])

  return (all_firings_x, all_firings_y)

def get_mean_firings(np_firings_x, np_firings_y):
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

  return (mean_firings_x, mean_firings_y);

def plot_connectivity_matrix(p, all_connections):
  output_name = '%sconnectivity_matrix_p=%.1f.eps' % (PLOT_OUTPUT_DIR, p)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(all_connections)
  fig.colorbar(cax)
  
  plt.title('Connections for p = %s' % p)
  plt.xlabel('To')
  plt.ylabel('From')
  plt.savefig(output_name, format='eps')

def plot_firings(p, all_firings_x, all_firings_y, mean_firings_x, mean_firings_y):
  output_name = '%sfirings_p=%.1f.eps' % (PLOT_OUTPUT_DIR, p)

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
  plt.xlim([0, T])
  plt.ylabel('Mean firing rate')
  plt.savefig(output_name, format='eps')

if __name__ == '__main__':
  main()

