from ModNetwork import *

import numpy as np
import numpy.random as rn

import sys

N_TRIALS = 20

SIM_TIME_MS = 1 * 1000

def main():
  for i in range(N_TRIALS):
    p = rn.rand()

    mn = ModNetwork(p)

    # Simulate
    run_net(mn)

    time_series = get_time_series(mn.net)
    print time_series

def run_net(mn):
  for t in xrange(SIM_TIME_MS):  
     sys.stdout.write('Simulating %s ms / %s ms\r' % (t, SIM_TIME_MS - 1))
     sys.stdout.flush()
   
     mn.update_with_poisson(0.01, t)
  
  # Empty line to prevent overwriting the last simulation line.
  print

def get_time_series(net):
  time_series = {}

  for layer in range(1, EXCIT_MODULES + 1):

    # Get total number of neurons which fired at each time point.
    sums = get_cumalative_firings(net.layer[layer].firings)

    time_series[layer] = []
    for i in np.arange(100, SIM_TIME_MS, 20):
      total_fired_in_range = sums[i] - sums[i - 50 - 1]
      mean = total_fired_in_range / 50.0

      time_series[layer].append(mean)

  return time_series

def get_cumalative_firings(firings):
  index = 0
  sums = []

  for t in range(SIM_TIME_MS):
    # Find the number of neurons which fired at the current time point.
    firings_at_t = 0
    while index < len(firings) and firings[index][0] == t:
      firings_at_t += 1
      index += 1

    total = firings_at_t + (0 if t == 0 else sums[t - 1])
    sums.append(total)

  return sums

if __name__ == "__main__":
  main()

