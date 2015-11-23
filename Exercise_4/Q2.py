from ModNetwork import *

from jpype import *

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

import sys

N_TRIALS = 20

SIM_TIME_MS = 1 * 1000

OUTPUT_FILE = 'result.txt'

def main():
  startJVM(getDefaultJVMPath(), '-Djava.class.path=../infodynamics.jar')

  calcClass = JPackage('infodynamics.measures.continuous.kraskov').MultiInfoCalculatorKraskov2
  calc = calcClass()

  ps = []
  ys = []

  # Clear the output file.
  open(OUTPUT_FILE, "w").close()

  for i in range(N_TRIALS):
    p = rn.rand()

    print
    print 'Trial', (i + 1), ', p =', p, ':'

    mn = ModNetwork(p)

    # Simulate
    run_net(mn)

    time_series = get_time_series(mn.net)

    # Ensure that there is a time series for each module and all time series
    # have the same length.
    assert len(time_series) == EXCIT_MODULES, 'incorrect number of time series'
    assert all(len(time_series[t]) == len(time_series[1]) for t in time_series), \
      'time series have different lengths'

    print 'Got', len(time_series), 'time series of length', len(time_series[1])

    calc.setProperty('PROP_NORMALISE', 'true')
    calc.initialise(len(time_series[1]))

    calc.startAddObservations()
    for t in time_series:
      java_series = JArray(JDouble, 1)(time_series[t])
      calc.addObservation(java_series)
    calc.finaliseAddObservations()

    result = calc.computeAverageLocalOfObservations()

    # Save the result in a file.
    output_file = open(OUTPUT_FILE, "a")
    output_file.write('%f, %f' % (p, result))
    output_file.write('\n')
    output_file.close()

    ps.append(p)
    ys.append(result)

  plt.scatter(ps, ys)
  plt.savefig('plots/multiinformation.eps', format='eps')

  shutdownJVM()

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
    sums = get_cumulative_firings(net.layer[layer].firings)

    time_series[layer] = []
    for i in np.arange(100, SIM_TIME_MS, 20):
      total_fired_in_range = sums[i] - sums[i - 50 - 1]
      mean = total_fired_in_range / 50.0

      time_series[layer].append(mean)

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

