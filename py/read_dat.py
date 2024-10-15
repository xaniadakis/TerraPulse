# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime

def ELA11C_ADCread(fn):
    f = open(fn,'rb')
    header = np.fromfile(f, np.uint8, count=64)

    first_sample = int(header[48])*256 + int(header[49])  
    tini = first_sample / 1250e3 
    tini = tini - 20e-6

    data = np.fromfile(f,np.uint8).astype(int)
    ld = len(data) - 64
    Bx = np.zeros(int(ld/4), dtype=int)
    By = np.zeros(int(ld/4), dtype=int)

    nr = 0
    i = 1
    for j in range(89):
        now1 = 256 * 256 * ((data[i]&12) //  4) + data[i+1] * 256 + data[i+2]
        now3 = 256 * 256 * (data[i]&3) + data[i+3] * 256 + data[i+4]
    
        Bx[nr] = now1
        By[nr] = now3
    
        nr = nr + 1
        i = i + 5
    
    i = i + 2

    for n in range(8836):
        #if i > ld:
        #    print('i = %d, n = %d; problem with the file => break\n' % (i,n)
        #    nr = nr - 1
        #    break
    
        for j in range(102):
            now1 = 256 * 256* ((data[i]&12) //  4) + data[i+1] * 256 + data[i+2]
            now3 = 256 * 256*(data[i]&3) + data[i+3] * 256 + data[i+4]
        
            Bx[nr] = now1
            By[nr] = now3
        
            nr = nr + 1
            i = i + 5
        
        i = i + 2         

    for j in range(82): 
        #if i>ld:
            #print('j = %d, i = %d\n' % (j,i))
            #break

        now1 = 256 * 256 * ((data[i]&12) //  4) + data[i+1] * 256 + data[i+2]
        now3 = 256 * 256 * (data[i]&3) + data[i+3] * 256 + data[i+4]

        Bx[nr] = now1
        By[nr] = now3

        nr = nr + 1    
        i = i + 5

    f.close()

    while Bx[nr] == 0 or By[nr] == 0:
        nr = nr-1

    print('Number of samples %d (fs=3kHz->901442|+1)' % nr)

    midADC = 2**18/2
    Bx = Bx[:nr+1] - midADC
    By = By[:nr+1] - midADC
    
    return(Bx, By, nr, tini)

def calibrate_HYL(Bx,By):
    a1_mVnT = 55.0; #[mV/nT] conversion coefficient from 13.12.2023
    a2_mVnT = 55.0; #[mV/nT] 

    a1 = a1_mVnT * 1e-3 / 1e3 #[V/pT]
    a2 = a2_mVnT * 1e-3 / 1e3 #[V/pT]
    ku = 4.26 #amplification in the receiver
    c1 = a1 * ku #system sensitivity
    c2 = a2 * ku #system sensitivity
    d = 2**18 #18-bit digital-to-analog converter
    V = 4.096*2 #[V] voltage range of digital-to-analog converter

    scale1 = c1*d/V 
    scale2 = c2*d/V    

    return -Bx/scale1, -By/scale2 #[pT] %the sign starting from 13/12/2023


yymm = '202301'
# input("year and month (e.g. 202409): ")
dd = 6
# int(input("day (e.g. 1): "))
hh1 = 0
# int(input("starting hour (e.g. 0): "))

fmin = 0 #Hz
fmax = 50 #Hz
folder = "C:\\Users\\echan\\Documents\\Parnon\\"

freq = 5e6/128/13 
M = int(20 * freq) #20-sec 
overlap = M//2

# for h in range(hh1,24):
#     for m in range(1):
#

h = hh1
m = 0
fn1 = "%s/%s%.2d/%s%.2d%.2d%.2d.dat" % (folder,yymm,dd,yymm,dd,h,10*m)
print(fn1)

HNS1, HEW1, nr, tini = ELA11C_ADCread(fn1)
HNS1, HEW1 = calibrate_HYL(HNS1,HEW1)


# fn1 = "%s/%s%.2d/%s%.2d%.2d%.2d.dat" % (folder,yymm,dd,yymm,dd,h,10*m+5)
# print(fn1)
#
# HNS2, HEW2, nr, tini = ELA11C_ADCread(fn1)
# HNS2, HEW2 = calibrate_HYL(HNS2,HEW2)
#
# HNS = np.append(HNS1, HNS2)
# HEW = np.append(HEW1, HEW2)
HNS = HNS1
HEW = HEW1

t = np.linspace(0,len(HNS)/freq,len(HNS))

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t,HNS,'r',lw=1,label=r'$B_{NS}$')
plt.plot(t,HEW,'b',lw=1,label=r'$B_{EW}$')
plt.ylabel("B [pT]")
plt.xlabel("Time [sec]")
plt.xlim([0,300])
plt.ylim([-200,0])
plt.grid(ls=':')
plt.legend()

F,S_NS = signal.welch(HNS, fs = freq, nperseg = M, noverlap = overlap, scaling='spectrum')
F,S_EW = signal.welch(HEW, fs = freq, nperseg = M, noverlap = overlap, scaling='spectrum')

S_NS = S_NS / (F[1] - F[0])
S_EW = S_EW / (F[1] - F[0])

S_NS = S_NS[(F>fmin) & (F<fmax)]
S_EW = S_EW[(F>fmin) & (F<fmax)]
F = F[(F>fmin) & (F<fmax)]

plt.subplot(2,1,2)
plt.plot(F,S_NS,'r',lw=1)
plt.plot(F,S_EW,'b',lw=1)
plt.ylabel(r"$PSD\ [pT^2/Hz]$")
plt.xlabel("Frequency [Hz]")
# plt.xlim([fmin,fmax])
plt.xlim([0,50])
plt.ylim([0, 0.6])
plt.grid(ls=':')

plt.tight_layout()
plt.savefig("%s%.2d_%.2d%.2d.jpg" % (yymm,dd,h,10*m), dpi=300)
plt.close()

# Downsampling to 100 Hz (1 sample every 10)
downsampling_factor = 30
new_freq = freq / downsampling_factor

# HNS_downsampled = HNS[::downsampling_factor]
# HEW_downsampled = HEW[::downsampling_factor]

# Ensure the length of HNS and HEW is a multiple of downsampling_factor
len_HNS = len(HNS) - (len(HNS) % downsampling_factor)
len_HEW = len(HEW) - (len(HEW) % downsampling_factor)

# Reshape and average
HNS_downsampled = np.mean(HNS1[:len_HNS].reshape(-1, downsampling_factor), axis=1).astype(int)
HEW_downsampled = np.mean(HEW1[:len_HEW].reshape(-1, downsampling_factor), axis=1).astype(int)

# Now HNS_downsampled and HEW_downsampled contain the averaged values

# Adjust the time vector accordingly
t_downsampled = np.linspace(0, len(HNS_downsampled) / (freq / downsampling_factor), len(HNS_downsampled))

plt.figure(figsize=(10, 6))

# Plotting downsampled data
plt.subplot(2, 1, 1)
plt.plot(t_downsampled, HNS_downsampled, 'r', lw=1, label=r'$B_{NS}$ Downsampled')
plt.plot(t_downsampled, HEW_downsampled, 'b', lw=1, label=r'$B_{EW}$ Downsampled')
plt.ylabel("B [pT]")
plt.xlabel("Time [sec]")
plt.xlim([0, 300])
plt.ylim([-200, 0])
plt.grid(ls=':')
plt.legend()

M = int(20 * new_freq) #20-sec
overlap = M//2

# Compute PSD for downsampled data
F_downsampled, S_NS_downsampled = signal.welch(HNS_downsampled, fs=new_freq, nperseg=M, noverlap=overlap, scaling='spectrum')
F_downsampled, S_EW_downsampled = signal.welch(HEW_downsampled, fs=new_freq, nperseg=M, noverlap=overlap, scaling='spectrum')

S_NS_downsampled = S_NS_downsampled / (F_downsampled[1] - F_downsampled[0])
S_EW_downsampled = S_EW_downsampled / (F_downsampled[1] - F_downsampled[0])

S_NS_downsampled = S_NS_downsampled[(F_downsampled > fmin) & (F_downsampled < fmax)]
S_EW_downsampled = S_EW_downsampled[(F_downsampled > fmin) & (F_downsampled < fmax)]
F_downsampled = F_downsampled[(F_downsampled > fmin) & (F_downsampled < fmax)]

plt.subplot(2, 1, 2)
plt.plot(F_downsampled, S_NS_downsampled, 'r', lw=1, label='PSD $B_{NS}$ Downsampled')
plt.plot(F_downsampled, S_EW_downsampled, 'b', lw=1, label='PSD $B_{EW}$ Downsampled')
plt.ylabel(r"$PSD\ [pT^2/Hz]$")
plt.xlabel("Frequency [Hz]")
plt.xlim([0, 50])
plt.ylim([0, 0.6])
plt.grid(ls=':')
plt.legend()

plt.tight_layout()
plt.savefig("%s%.2d_%.2d%.2d_downscale.jpg" % (yymm, dd, h, 10*m), dpi=300)
plt.close()
