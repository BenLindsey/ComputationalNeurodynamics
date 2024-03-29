from ModNetwork import *

from jpype import *

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

import os
#import multiprocessing as mp

N_TRIALS = 48

IGNORE_MS = 1000 # Ignore the first milliseconds of each simulation.
SIM_TIME_MS = 60 * 1000

OUTPUT_DIR = 'results/'
SERIES_FILE = OUTPUT_DIR + 'series_'
P_FILE = OUTPUT_DIR + 'p_'

def main():
  (p_values, all_time_series) = get_data()

  startJVM(getDefaultJVMPath(), '-Djava.class.path=../infodynamics.jar')

  calcClass = JPackage('infodynamics.measures.continuous.kraskov').MultiInfoCalculatorKraskov2
  calc = calcClass()

  ys = []
  for i in range(N_TRIALS):
    p = p_values[i]
    time_series = all_time_series[i]

    calc.setProperty('PROP_NORMALISE', 'true')
    calc.setProperty('K', '4')
    calc.initialise(8)

    calc.setObservations(np.transpose(time_series))

    result = calc.computeAverageLocalOfObservations() * np.log2(np.e)

    ys.append(result)

  plt.scatter(p_values, ys)
  plt.ylabel('Integration (bits)')
  plt.xlabel('Rewiring probability p')
  plt.xlim([0, 1])
  plt.savefig('plots/multiinformation.eps', format='eps')
  plt.show()

  shutdownJVM()

def get_data():
  if all_result_files_exist():
    print 'Using saved time series.'

    p_files = [P_FILE + str(i) + '.txt' for i in range(N_TRIALS)]
    series_files = [SERIES_FILE + str(i) + '.txt' for i in range(N_TRIALS)]

    p_values = [np.loadtxt(f) for f in p_files]
    all_time_series = [np.loadtxt(f) for f in series_files]

  else:
    print 'Calculating time series.'

    # Get the time series
    p_values = [rn.rand() for i in range(N_TRIALS)]
  #  pool = mp.Pool(8)
    all_time_series = map(get_time_series_from_p, p_values)

    # Ensure that the directory for the results exists.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save to files
    for t in range(len(all_time_series)):
      np.savetxt(SERIES_FILE + str(t) + ".txt", all_time_series[t])
      np.savetxt(P_FILE + str(t) + ".txt", [p_values[t]])

  return (p_values, all_time_series)

def all_result_files_exist():
  p_files = [P_FILE + str(i) + '.txt' for i in range(N_TRIALS)]
  series_files = [SERIES_FILE + str(i) + '.txt' for i in range(N_TRIALS)]

  return all(os.path.isfile(f) for f in p_files + series_files)

def get_time_series_from_p(p):
  mn = ModNetwork(p)
  run_net(mn)

  print 'Finished a simulation.'
  return get_time_series(mn.net)

def run_net(mn):
  for t in xrange(SIM_TIME_MS):  
     mn.update_with_poisson(0.01, t)

     if t % 20000 == 0:
       print 'Simulation at time', t

def get_time_series(net):
  time_series = []

  for layer in range(1, EXCIT_MODULES + 1):
    firings = net.layer[layer].firings[IGNORE_MS:]
    firing_times = [f[0] for f in firings]

    firings_per_t = [0] * SIM_TIME_MS
    for t in firing_times:
      firings_per_t[t] += 1

    sums = np.cumsum(np.insert(firings_per_t, 0, 0))
    means = (sums[50:] - sums[:-50]) / 50.0

    time_series.append(means[0::20])

  # Ensure that there is a time series for each module and all time series
  # have the same length.
  assert len(time_series) == EXCIT_MODULES, 'incorrect number of time series'
  assert all(len(time_series[t]) == len(time_series[1]) for t in range(len(time_series))), \
    'time series have different lengths'

  return time_series

if __name__ == "__main__":
  main()

