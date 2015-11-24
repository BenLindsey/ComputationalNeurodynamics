from ModNetwork import *

from jpype import *

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

import sys
import multiprocessing as mp

N_TRIALS = 20

SIM_TIME_MS = 60 * 1000

SERIES_FILE = 'series_'
P_FILE = 'p_'

def main():
  startJVM(getDefaultJVMPath(), '-Djava.class.path=../infodynamics.jar')

  calcClass = JPackage('infodynamics.measures.continuous.kraskov').MultiInfoCalculatorKraskov2
  calc = calcClass()

  # Get the time series
  p_values = [rn.rand() for i in range(N_TRIALS)]
  pool = mp.Pool(8)
  all_time_series = pool.map(get_time_series_from_p, p_values)

  # Save to files
  for t in range(len(all_time_series)):
    np.savetxt(SERIES_FILE + str(t) + ".txt", all_time_series[t])
    np.savetxt(P_FILE + str(t) + ".txt", [p_values[t]])

  ys = []
  for i in range(N_TRIALS):
    p = p_values[i]
    time_series = all_time_series[i]

    calc.setProperty('PROP_NORMALISE', 'true')
    calc.setProperty('K', '4')
    calc.initialise(len(time_series[1]))

    java_time_series = JArray(JDouble, 2)(time_series)
    calc.setObservations(java_time_series)

    result = calc.computeAverageLocalOfObservations()

    ys.append(result)

  plt.scatter(p_values, ys)
  plt.savefig('plots/multiinformation.eps', format='eps')
  plt.show()

  shutdownJVM()

def get_time_series_from_p(p):
  mn = ModNetwork(p)
  run_net(mn)

  print 'Finished a simulation.'
  return get_time_series(mn.net)

def run_net(mn):
  for t in xrange(SIM_TIME_MS):  
     mn.update_with_poisson(0.01, t)

     if t % 1000 == 0:
       print 'Simulation at time', t

def get_time_series(net):
  time_series = []

  for layer in range(1, EXCIT_MODULES + 1):

    # Get total number of neurons which fired at each time point.
    sums = get_cumulative_firings(net.layer[layer].firings)

    layer_time_series = []
    for i in np.arange(1000, SIM_TIME_MS, 20):
      total_fired_in_range = sums[i] - sums[i - 50 - 1]
      mean = total_fired_in_range / 50.0

      layer_time_series.append(mean)
    time_series.append(layer_time_series)

  # Ensure that there is a time series for each module and all time series
  # have the same length.
  assert len(time_series) == EXCIT_MODULES, 'incorrect number of time series'
  assert all(len(time_series[t]) == len(time_series[1]) for t in range(len(time_series))), \
    'time series have different lengths'

  return time_series

def get_cumulative_firings(firings):
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

