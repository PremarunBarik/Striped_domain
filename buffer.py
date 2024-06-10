import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycorrelate as pyc

file_path = "/Users/premarunbarik/Documents/Research/Data_Codes/StripedDomain_DetectorData/qr2_250K_timestamps.csv"
df = pd.read_csv(file_path)
timestamps = df.to_numpy()
timestamps =  np.round( np.array(timestamps)/5120, 0)

tt= timestamps[1:10000]
binwidth = 1
bins_tt = np.arange(tt.min(), tt.max(), binwidth) 

tx, _ = np.histogram(tt, bins=bins_tt)
C = np.correlate(tx, tx, mode='full')

Gn = C[tx.size:(tx.size+4000)]
Gn_t = np.arange(1, Gn.size+1)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(Gn_t, Gn, alpha=1, lw=2, label='numpy.correlate')
plt.xscale('log')
plt.legend(loc='best', fontsize='x-large');

#plt.show()

np.savetxt('Gn_t.txt', (Gn_t+1)*5120/1e8)
np.savetxt('Gn.txt', Gn)

