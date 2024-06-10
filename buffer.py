import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycorrelate as pyc

file_path = "/Users/premarunbarik/Documents/Research/Data_Codes/StripedDomain_DetectorData/qr2_250K_timestamps.csv"
df = pd.read_csv(file_path)
timestamps = df.to_numpy()
timestamps =  np.round( np.array(timestamps)/5120, 0)

tt= timestamps
#tt = np.array(sorted(tt))
tt = np.array(tt)

tau_array1 = np.arange(1,10,1)
tau_array2 = np.arange(10,4001,10)
tau_array = np.concatenate([tau_array1, tau_array2])

lagbins = np.array(tau_array)
G = pyc.pcorrelate(tt, tt, lagbins, normalize = False)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(lagbins[2:]*5120/1e8, G[1:], alpha=1, lw=2, label='numpy.pycorrelate')
plt.xscale('log')
plt.legend(loc='best', fontsize='x-large');

plt.show()

np.savetxt('Gp_t_unsrt.txt', (lagbins[2:] )*5120/1e8)
np.savetxt('Gp_unsrt.txt', G[1:])

