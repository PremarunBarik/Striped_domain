import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import pycorrelate as pyc

url = 'http://files.figshare.com/2182601/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5'
pyc.utils.download_file(url, save_dir='data')

fname = './data/' + url.split('/')[-1]
h5 = h5py.File(fname)
unit = 12.5e-9

num_ph = int(3e6)
detectors = h5['photon_data']['detectors'][:num_ph]
timestamps = h5['photon_data']['timestamps'][:num_ph]
t = timestamps[detectors == 0]
u = timestamps[detectors == 1]

bins = pyc.make_loglags(4, 10, 10)

G = pyc.pcorrelate(t, t, bins, normalize=True)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(bins[1:], G, drawstyle='steps-pre') 
plt.xlabel('Time (s)')
#for x in bins[1:]: plt.axvline(x*unit, lw=0.2) # to mark bins 
plt.grid(True); plt.grid(True, which='minor', lw=0.3) 
plt.xscale('log')
#plt.xlim(30e-9, 2)

plt.show()

G = pyc.pcorrelate(t, u, bins, normalize=True)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(bins[1:]*unit, G, drawstyle='steps-pre') 
plt.xlabel('Time (s)')
#for x in bins[1:]: plt.axvline(x*unit, lw=0.2) # to mark bins 
plt.grid(True); plt.grid(True, which='minor', lw=0.3) 
plt.xscale('log')
plt.xlim(30e-9, 2)

plt.show()

################################################################

tt = t[:5000]
binwidth = 50e-6
bins_tt = np.arange(0, tt.max()*unit, binwidth) / unit
maxlag_sec = 3.9
lagbins = (np.arange(0, maxlag_sec, binwidth) / unit).astype('int64')
Gp = pyc.pcorrelate(tt, tt, lagbins, normalize=True) #* int(binwidth / unit)

plt.plot(lagbins[2:]* unit * 1e3, Gp[1:], '-' , label = 'pycorrelate.pcorrelate')
plt.xlabel('Time delay (ms)', fontsize='large')
plt.ylabel('g2', fontsize='large')
plt.xscale('log')
plt.show()


################################################################

import numpy as np
import pycorrelate

# Generate example photon arrival times (sorted for demonstration purposes)
timestamps = np.sort(np.random.uniform(0, 10, 1000)) *1e5

tau_array1 = np.arange(1,10,1)
tau_array2 = np.arange(10,1000,10)
tau_array = np.concatenate([tau_array1, tau_array2])
tau = tau_array[:]

duration = max(timestamps)-min(timestamps)                                                                                                                         
idtau = ( duration - tau > 0 )
tau_mod = tau[idtau]   

# Calculate autocorrelation using pycorrelate
autocorr = pycorrelate.pcorrelate(timestamps, timestamps, tau_mod, normalize=True)

# Plot autocorrelation function
import matplotlib.pyplot as plt
plt.plot(autocorr)
plt.xlabel('Time Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Photon Timestamps')
plt.show()
