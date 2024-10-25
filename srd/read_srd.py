import os
import sys
import time
import datetime
import struct
import array
import matplotlib.pyplot as plt
from scipy import signal

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
if path=='':
	sys.exit()
root.destroy()
root.mainloop()

size = os.path.getsize(path)

f = open(path, 'rb')

f.seek(0x8)
sec = int.from_bytes(bytes(f.read(1)), 'little')
min = int.from_bytes(bytes(f.read(1)), 'little')
hour = int.from_bytes(bytes(f.read(1)), 'little')
day = int.from_bytes(bytes(f.read(1)), 'little')
date = int.from_bytes(bytes(f.read(1)), 'little')
month = int.from_bytes(bytes(f.read(1)), 'little')
year = int.from_bytes(bytes(f.read(1)), 'little')
dt = datetime.datetime(year+1970, month, date, hour, min, sec)

f.seek(0x0f)
fs = struct.unpack('f', f.read(4))[0]

f.seek(0x13)
ch = int.from_bytes(bytes(f.read(1)), 'little')

f.seek(0x210)
vals = array.array('H')
num = int((size-0x210)/2)	# 2-bytes/sample
vals.fromfile(f, num)
f.close()

x=None
y=None
t=None

MAX_VAL = 32767.0

if ch==0:	# 1 channel
	x = array.array('d',[0]*num)
	t = array.array('d',[0]*num)
	k = 0
	for sample in vals:
		x[k]=float(sample)*4.096/MAX_VAL - 2.048
		t[k]=float(k)/fs
		k+=1
elif ch==1:	# 2 channels
	x = array.array('d',[0]*int(num/2))
	y = array.array('d',[0]*int(num/2))
	t = array.array('d',[0]*int(num/2))
	i = 0
	for sample in vals:
		k=i//2
		if i%2==0:
			x[k]=float(sample)*4.096/MAX_VAL - 2.048
		else:
			y[k]=float(sample)*4.096/MAX_VAL - 2.048
		t[k]=float(k)/fs
		i+=1

print(dt)			# date
print(ch)			# channels 0 = 1ch (x=N-S), 1 = 2ch (x=N-S, y=E-W)
print(fs)			# sampling rate (Hz)


plt.figure(1)
plt.plot(t,x, linewidth=0.5)
plt.title(dt.strftime("%d/%m/%Y %H:%M:%S")+' (N-S)')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (sec)')
plt.grid()
plt.show(block = False)

nseg = 20*fs
q=0
ff, Pxx = signal.welch(x, fs, nperseg=nseg, noverlap=nseg/2)
for index, item in enumerate(ff.tolist()):
    if item >= 42:
        q=index
        break
plt.figure(2)
plt.plot(ff[0:q],Pxx[0:q], linewidth=0.5)
plt.title(dt.strftime("%d/%m/%Y %H:%M:%S")+' (N-S)')
plt.ylabel('Amplitude (a.u.)')
plt.xlabel('Spectrum (Hz)')
plt.grid()


if ch==1:
	plt.show(block = False)
	plt.figure(3)
	plt.plot(t,y, linewidth=0.5)
	plt.title(dt.strftime("%d/%m/%Y %H:%M:%S")+' (E-W)')
	plt.ylabel('Amplitude (V)')
	plt.xlabel('Time (sec)')
	plt.grid()
	plt.show(block = False)
	
	nseg = 20*fs
	q=0
	ff, Pxx = signal.welch(y, fs, nperseg=nseg, noverlap=nseg/2)
	for index, item in enumerate(ff.tolist()):
		if item >= 42:
			q=index
			break
	plt.figure(4)
	plt.plot(ff[0:q],Pxx[0:q], linewidth=0.5)
	plt.title(dt.strftime("%d/%m/%Y %H:%M:%S")+' (E-W)')
	plt.ylabel('Amplitude (a.u.)')
	plt.xlabel('Spectrum (Hz)')
	plt.grid()
	plt.show()
else:
	plt.show()


