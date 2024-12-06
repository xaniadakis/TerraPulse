from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import struct
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
import zstandard as zstd
import time
from scipy.interpolate import interp1d
from scipy.signal import welch, windows
from scipy.optimize import curve_fit

new_logger_fact = 0.25  # New logger amp factor
old_logger = 0  # Global variable in the original code; define as needed

def get_srd_info(fn):
    """
    Extracts metadata from an SRD file.

    Parameters:
    fn (str): The file name (path) of the SRD data file.

    Returns:
    tuple:
        - date (float): The timestamp in seconds since the epoch, corrected if necessary.
        - fs (float): Sampling frequency in Hz.
        - ch (int): Channel information (0 or 1).
        - vbat (float): Battery voltage in volts.
        - ok (int): Success flag (1 if successful, 0 if not).
    """
    ok = 0
    fs = -1
    ch = 0
    date = 0
    vbat = 0.0
    DATALOGGERID = int("CAD0FFE51513FFDC", 16)

    # Check file size
    if os.path.getsize(fn) < (2 * 512):
        return date, fs, ch, vbat, ok

    with open(fn, 'rb') as fp:
        # Read DATALOGGERID
        ID = struct.unpack('Q', fp.read(8))[0]
        if ID != DATALOGGERID:
            print(f'File "{fn}" is not a logger record!')
            return date, fs, ch, vbat, ok

        # Read timestamp components
        S = struct.unpack('B', fp.read(1))[0]
        MN = struct.unpack('B', fp.read(1))[0]
        H = struct.unpack('B', fp.read(1))[0]
        DAY = struct.unpack('B', fp.read(1))[0]
        D = struct.unpack('B', fp.read(1))[0]
        M = struct.unpack('B', fp.read(1))[0]
        Y = struct.unpack('B', fp.read(1))[0] + 1970

        # Convert to datetime
        date = datetime(Y, M, D, H, MN, S)

        # Define correction dates and adjust date if necessary
        t0 = datetime(2016, 1, 1)
        t1 = datetime(2017, 8, 1)
        t2 = datetime(2018, 8, 1)

        if t0 < date < t1:
            tslop = 480 / 600  # seconds-offset per day
            days_diff = (date - t0).days
            dt_seconds = days_diff * tslop
            date -= timedelta(seconds=dt_seconds)

        # Set to timestamp
        date = date.timestamp()

        # Read fs
        fp.seek(15, os.SEEK_SET)
        fs = struct.unpack('f', fp.read(4))[0]

        # Read ch
        fp.seek(19, os.SEEK_SET)
        ch = struct.unpack('B', fp.read(1))[0]

        # Read vbat
        fp.seek(20, os.SEEK_SET)
        vbat = struct.unpack('f', fp.read(4))[0]

        # Successfully read info
        ok = 1

    return date, fs, ch, vbat, ok

def read_srd_file(fn):
    """
    Reads data from an SRD file and processes it for analysis.

    Parameters:
    fn (str): The file name (path) of the SRD data file.

    Returns:
    tuple:
        - t (float): Timestamp of the data file in seconds since the epoch.
        - fs (float): Sampling frequency in Hz.
        - x (numpy.ndarray): Array of processed data samples for channel 0 (or X data for channel 1).
        - y (numpy.ndarray): Array of processed data samples for channel 1 (empty if only one channel exists).
    """
    # Initializations
    x = []
    y = []

    # Call get_srd_info to extract metadata
    t, fs, ch, vb, ok = get_srd_info(fn)
    print(f"Battery voltage: {vb}\n")
    if not ok or fs <= 0:
        return t, fs, x, y

    # Read the file after skipping the header (512 + 16 bytes)
    with open(fn, 'rb') as f:
        f.seek(512 + 16)
        x = np.fromfile(f, dtype=np.uint16).astype(float)

    # Define threshold date for MAX_VAL determination
    date1 = datetime(2017, 8, 10).timestamp()
    if t < date1:
        MAX_VAL = 65535.0
    else:
        MAX_VAL = 32767.0
        if np.any(x[:10000] > MAX_VAL):  # Detect faulty shift
            with open(fn, 'rb') as f:
                f.seek(512 + 17)
                x = np.fromfile(f, dtype=np.uint16).astype(float)

    # Process data length to be even
    N = len(x)
    if N % 2 != 0:
        x = x[:-1]
        N -= 1

    # Process data based on channel info
    if ch == 0:
        x = x * 4.096 / MAX_VAL - 2.048  # Scale samples (x->Volt)
    else:
        xx = x.reshape((N // 2, 2)).T
        x = xx[0, :] * 4.096 / MAX_VAL - 2.048  # Scale x
        y = xx[1, :] * 4.096 / MAX_VAL - 2.048  # Scale y

    # Remove DC offset
    x -= np.mean(x)
    if ch == 1:
        y -= np.mean(y)

    return t, fs, x, y

# Get PSD of the time-domain NS & EW signals
def compute_PSD(HNS, HEW, frequency, fmin, fmax):
    M = int(20 * frequency)  # 20-sec
    overlap = M // 2
    S_EW = []

    f, S_NS = signal.welch(HNS, fs=frequency, nperseg=M, noverlap=overlap, scaling='spectrum')
    S_NS = S_NS / (f[1] - f[0])
    S_NS = S_NS[(f > fmin) & (f < fmax)]

    if len(HEW)>0:
        f, S_EW = signal.welch(HEW, fs=frequency, nperseg=M, noverlap=overlap, scaling='spectrum')
        S_EW = S_EW / (f[1] - f[0])
        S_EW = S_EW[(f > fmin) & (f < fmax)]

    f = f[(f > fmin) & (f < fmax)]

    return S_NS, S_EW, f

def plot_PSD(f, S_NS, S_EW):
    # Plot PSD
    # plt.subplot(2, 1, 2)
    plt.plot(f, S_NS, 'r', lw=1, label='PSD $B_{NS}$')
    if len(S_EW)>0:
        plt.plot(f, S_EW, 'b', lw=1, label='PSD $B_{EW}$')
    plt.ylabel(r"$PSD\ [pT^2/Hz]$")
    plt.xlabel("Frequency [Hz]")
    plt.xlim([0, 50])
    # plt.ylim([0, 0.6])
    plt.grid(ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_signal(HNS, HEW, frequency, date):
    t = np.linspace(0, len(HNS) / frequency, len(HNS))
    plt.figure(figsize=(10, 6))
    plt.title(datetime.fromtimestamp(date))
    plt.plot(t, HNS, 'r', lw=1, label=r'$B_{NS}$')
    if len(HEW)>0:
        plt.plot(t, HEW, 'b', lw=1, label=r'$B_{EW}$')
    plt.ylabel("B [V]")
    plt.xlabel("Time [sec]")
    plt.xlim([0, 600])
    # plt.ylim([-200, 0])
    plt.grid(ls=':')
    plt.legend()
    plt.show()

def decimate_signal(HNS, HEW, downsampling_factor):
    len_HNS = len(HNS) - (len(HNS) % downsampling_factor)
    HNS_downsampled = np.mean(HNS[:len_HNS].reshape(-1, downsampling_factor), axis=1)

    HEW_downsampled = []
    if len(HEW)>0:
        len_HEW = len(HEW) - (len(HEW) % downsampling_factor)
        HEW_downsampled = np.mean(HEW[:len_HEW].reshape(-1, downsampling_factor), axis=1)

    return HNS_downsampled, HEW_downsampled

def get_equalizer(freqs, date):
    # Define important dates
    date1 = datetime.strptime('12-Feb-2017', '%d-%b-%Y')
    date2 = datetime.strptime('10-Aug-2017', '%d-%b-%Y')
    date3 = datetime.strptime('03-Dec-2018', '%d-%b-%Y')
    date4 = datetime.strptime('01-Jul-2019', '%d-%b-%Y')

    # Select appropriate response based on the date
    if date < date1:
        res_amp, res_coil = get_res_1()
    elif date1 <= date < date2:
        res_amp, res_coil = get_res_2()
    elif date2 <= date < date3:
        res_amp, res_coil = get_res_3()
    elif date3 <= date < date4:
        res_amp, res_coil = get_res_4()
    elif date >= date4:
        if old_logger != 0:
            sreq1 = np.ones(len(freqs)) * 40000
            sreq2 = sreq1
            return sreq1, sreq2
        else:
            res_amp, res_coil = get_res_5()

    sreq1 = get_sreq(res_amp, res_coil, freqs)
    sreq2 = sreq1
    return sreq1, sreq2

def get_sreq(res_amp, res_coil, freqs):
    if res_amp is None or res_coil is None:
        return 1

    # For amps
    f_amp = res_amp[:, 0]
    mag_amp = res_amp[:, 1]
    interp_amp = interp1d(f_amp, mag_amp, kind='linear', fill_value='extrapolate')
    sreq = interp_amp(freqs)

    # For coil
    f_coil = res_coil[:, 0]
    mag_coil = res_coil[:, 1]
    interp_coil = interp1d(f_coil, mag_coil, kind='linear', fill_value='extrapolate')
    sreqc = interp_coil(freqs) * 1e-9  # Convert to Volt/pTesla

    sreq = sreq * sreqc
    sreq = sreq ** 2  # Power
    sreq = 1.0 / sreq  # Inverse
    return sreq

def get_res_1():
    res_amp = np.array([[
        1, 1341.44364189815], [
        2, 6560.35398044948], [
        3, 16099.4157858125], [
        4, 23620.2450875275], [
        5, 27817.3820718944], [
        6, 29932.8964366596], [
        7, 30945.6738337945], [
        8, 31438.3593884841], [
        9, 31668.6977306129], [
        10, 31954.8946897702], [
        11.001, 32392.1400440383], [
        12, 33013.0702887598], [
        13, 33767.6846394033], [
        14.001, 34723.3481696889], [
        15.002, 35826.5035539261], [
        16, 37080.4981249706], [
        17.001, 38329.6809050351], [
        18, 39397.2708776811], [
        19.002, 40000], [
        20.002, 39907.9483464699], [
        21.001, 38893.4972829072], [
        22.004, 37029.2420906186], [
        23.002, 34579.2036485928], [
        24, 31683.1331035529], [
        25.004, 28563.0004654885], [
        26.003, 25445.7967436728], [
        27, 22402.8619695916], [
        28.005, 19271.6412914429], [
        29.004, 16355.4867492691], [
        30.003, 13626.3644304042], [
        31.004, 11124.442329116], [
        32, 8965.20342892409], [
        33.003, 7107.01527743636], [
        34.007, 5581.049911871], [
        35.009, 4368.89700152199], [
        36, 3411.1413882017], [
        37.012, 2650.25078845379], [
        38.004, 2080.36736978091], [
        39.01, 1648.97984801017], [
        40.012, 1308.80714655565], [
        41.006, 1058.59401559648], [
        42.002, 868.423667735373], [
        43.015, 752.103851001846], [
        44.008, 673.023112287327], [
        45.011, 637.666908999618], [
        46.003, 574.067584742438], [
        47, 563.188752961605], [
        48, 563.188752961605], [
        49.021, 535.154840295611], [
        50.019, 572.603126618095], [
        51.011, 565.280835996381], [
        52.019, 487.455347102727], [
        53.016, 539.339006365163], [
        54, 523.22996699739], [
        55.021, 533.899590474746], [
        56.024, 515.48925976872], [
        57.006, 516.535301286108], [
        58.023, 533.062757260836], [
        59.014, 529.087799494762], [
        60.007, 537.456131633865], [
        61.033, 524.694425121733], [
        62.026, 528.878591191284], [
        63.015, 556.912503857278], [
        64, 525.322050032166], [
        65.016, 518.418176017406], [
        66.025, 522.811550390435], [
        67.025, 512.560343520034], [
        68.015, 530.343049315627], [
        69.034, 535.364048599089], [
        70.039, 550.217838145996], [
        71.029, 509.422218967871], [
        72, 470.718682824521], [
        73.046, 464.442433720194], [
        74.024, 515.280051465243], [
        75.028, 507.539344236573], [
        76.008, 501.681511739201], [
        77.014, 507.120927629618], [
        78.047, 518.418176017406], [
        79.051, 519.673425838272], [
        80.023, 501.263095132246], [
        81.02, 503.982803077454], [
        82.042, 510.049843878303], [
        83.027, 507.957760843528], [
        84.036, 512.141926913079], [
        85.005, 551.263879663384], [
        86.064, 516.953717893063], [
        87.012, 506.284094415707], [
        88.051, 542.895547524281], [
        89.043, 499.589428704425], [
        90.059, 521.347092266092], [
        91.022, 556.912503857278], [
        92.007, 534.108798778224], [
        93.013, 499.380220400948], [
        94.041, 517.372134500018], [
        95.01, 502.099928346156], [
        96, 522.602342086957], [
        97.011, 494.777637724441], [
        98.043, 510.049843878303], [
        99.008, 509.422218967871], [
        100.083, 495.196054331396]
    ])

    res_coil = get_res_coil(12.45)
    return res_amp, res_coil

def get_res_coil(s):
    # f = freq vector, s = sensitivity (nVolt/pTesla/Hz)
    f = np.arange(1, 101).reshape(-1, 1)
    m = f * s
    out = np.hstack((f, m))
    return out

def get_res_2():
    res_amp = np.array([[
        1, 8085.7], [
        1.5, 16513], [
        2, 24561], [
        2.5, 29952], [
        3, 32959], [
        3.5, 34595], [
        4, 35550], [
        4.5, 36122], [
        5, 36527], [
        5.5, 36777], [
        6, 37001], [
        6.5, 37152], [
        7, 37263], [
        7.5, 37421], [
        8, 37512], [
        8.5, 37639], [
        9, 37712], [
        9.5, 37840], [
        10, 37937], [
        10.5, 38030], [
        11, 38125], [
        11.5, 38196], [
        12, 38292], [
        12.5, 38384], [
        13, 38493], [
        13.5, 38577], [
        14, 38654], [
        14.5, 38782], [
        15, 38846], [
        15.5, 38927], [
        16, 39016], [
        16.5, 39121], [
        17, 39185], [
        17.5, 39264], [
        18, 39334], [
        18.5, 39403], [
        19, 39485], [
        19.5, 39512], [
        20, 39573], [
        20.5, 39599], [
        21, 39630], [
        21.5, 39645], [
        22, 39647], [
        22.5, 39624], [
        23, 39595], [
        23.5, 39550], [
        24, 39487], [
        24.5, 39392], [
        25, 39294], [
        25.5, 39166], [
        26, 39007], [
        26.5, 38791], [
        27, 38576], [
        27.5, 38302], [
        28, 37994], [
        28.5, 37658], [
        29, 37239], [
        29.5, 36815], [
        30, 36307], [
        30.5, 35764], [
        31, 35130], [
        31.5, 34460], [
        32, 33731], [
        32.5, 32903], [
        33, 32011], [
        33.5, 31050], [
        34, 29997], [
        34.5, 28868], [
        35, 27665], [
        35.5, 26398], [
        36, 25069], [
        36.5, 23638], [
        37, 22163], [
        37.5, 20616], [
        38, 19062], [
        38.5, 17458], [
        39, 15864], [
        39.5, 14269], [
        40, 12723], [
        40.5, 11181], [
        41, 9697.3], [
        41.5, 8295.8], [
        42, 6993.8], [
        42.5, 5825.9], [
        43, 4745], [
        43.5, 3821], [
        44, 3014.1], [
        44.5, 2326.9], [
        45, 1782.7], [
        45.5, 1375.2], [
        46, 1071.7], [
        46.5, 928.86], [
        47, 841.76], [
        47.5, 808.7], [
        48, 797.48], [
        48.5, 792.77], [
        49, 816.66], [
        49.5, 787.66], [
        50, 812.52], [
        50.5, 820.91], [
        51, 814.52], [
        51.5, 813.18], [
        52, 820.64], [
        52.5, 820.52], [
        53, 804.37], [
        53.5, 806.05], [
        54, 795.45], [
        54.5, 810.25], [
        55, 809.4], [
        55.5, 852.89], [
        56, 886.46], [
        56.5, 949.25], [
        57, 1008.5], [
        57.5, 1119.4], [
        58, 1225.5], [
        58.5, 1412.8], [
        59, 1596.5], [
        59.5, 1800], [
        60, 2017.3], [
        60.5, 2259.6], [
        61, 2518], [
        61.5, 2775.8], [
        62, 3050.1], [
        62.5, 3328.3], [
        63, 3612.9], [
        63.5, 3896], [
        64, 4193.7], [
        64.5, 4521.4], [
        65, 4819.3], [
        65.5, 5141.8], [
        66, 5450.3], [
        66.5, 5752], [
        67, 6067.6], [
        67.5, 6367.2], [
        68, 6691.7], [
        68.5, 7016], [
        69, 7333.5], [
        69.5, 7626.3], [
        70, 7933.5], [
        70.5, 8249.1], [
        71, 8555.7], [
        71.5, 8838.6], [
        72, 9155.2], [
        72.5, 9436.4], [
        73, 9739], [
        73.5, 10019], [
        74, 10302], [
        74.5, 10598], [
        75, 10860], [
        75.5, 11138], [
        76, 11399], [
        76.5, 11661], [
        77, 11913], [
        77.5, 12183], [
        78, 12446], [
        78.5, 12687], [
        79, 12940], [
        79.5, 13195], [
        80, 13403], [
        80.5, 13665], [
        81, 13876], [
        81.5, 14082], [
        82, 14348], [
        82.5, 14543], [
        83, 14759], [
        83.5, 14982], [
        84, 15193], [
        84.5, 15396], [
        85, 15603], [
        85.5, 15803], [
        86, 15983], [
        86.5, 16208], [
        87, 16368], [
        87.5, 16559], [
        88, 16760], [
        88.5, 16925], [
        89, 17102], [
        89.5, 17282], [
        90, 17450], [
        90.5, 17593], [
        91, 17792], [
        91.5, 17958], [
        92, 18122], [
        92.5, 18259], [
        93, 18425], [
        93.5, 18575], [
        94, 18735], [
        94.5, 18885], [
        95, 19021], [
        95.5, 19160], [
        96, 19311], [
        96.5, 19444], [
        97, 19584], [
        97.5, 19737], [
        98, 19842], [
        98.5, 19971], [
        99, 20105], [
        99.5, 20222], [
        100, 20366]
    ])
    res_amp[:, 1] = res_amp[:, 1] / 4.5  # Convert res_amp values as per your request
    res_coil = get_res_coil(12.45)
    return res_amp, res_coil

def get_res_3():
    res_amp = np.array([[
        1, 7536.69727790345], [
        1.5, 14075.804128733], [
        2, 20788.4330991098], [
        2.5, 25464.2899619352], [
        3, 28224.2548357119], [
        3.5, 29817.038110306], [
        4, 30729.6827856485], [
        4.5, 31347.7766658598], [
        5, 31731.8251278891], [
        5.5, 32031.3435341674], [
        6, 32246.6913078276], [
        6.5, 32414.4490790816], [
        7, 32579.8405344774], [
        7.5, 32731.7573654057], [
        8, 32907.5635166288], [
        8.5, 33002.4451191577], [
        9, 33141.6445541604], [
        9.5, 33273.1183488773], [
        10, 33423.2335526053], [
        10.5, 33531.6231836318], [
        11, 33687.2723859913], [
        11.5, 33836.1969364187], [
        12, 33969.7145811141], [
        12.5, 34107.6906557404], [
        13, 34289.2709312872], [
        13.5, 34425.1136719645], [
        14, 34582.4861999387], [
        14.5, 34757.6327522452], [
        15, 34905.3638553314], [
        15.5, 35079.761934586], [
        16, 35252.3665260725], [
        16.5, 35425.7795722515], [
        17, 35597.7458343771], [
        17.5, 35781.1678555634], [
        18, 35972.4248579846], [
        18.5, 36166.3360557802], [
        19, 36342.739316488], [
        19.5, 36556.1865825384], [
        20, 36720.0489572528], [
        20.5, 36913.7864640481], [
        21, 37112.6883222693], [
        21.5, 37313.4826171192], [
        22, 37496.1484240811], [
        22.5, 37676.5936097747], [
        23, 37871.1430747031], [
        23.5, 38058.7934686306], [
        24, 38259.9518163822], [
        24.5, 38432.868743018], [
        25, 38618.2226783232], [
        25.5, 38786.7671368896], [
        26, 38955.3583536307], [
        26.5, 39121.0308061177], [
        27, 39275.4871087431], [
        27.5, 39416.028069012], [
        28, 39547.751491648], [
        28.5, 39664.9019083039], [
        29, 39772.9504983706], [
        29.5, 39868.4359807986], [
        30, 39930.6201591755], [
        30.5, 39976.3015767679], [
        31, 40000], [
        31.5, 39997.7392167325], [
        32, 39950.7724222653], [
        32.5, 39919.8015409194], [
        33, 39831.7258041821], [
        33.5, 39694.4880283177], [
        34, 39515.0237810925], [
        34.5, 39303.5899292695], [
        35, 39040.0745459969], [
        35.5, 38699.4082737883], [
        36, 38323.010548112], [
        36.5, 37877.0336899814], [
        37, 37361.2986132219], [
        37.5, 36754.1903714176], [
        38, 36075.1518711734], [
        38.5, 35306.0872445579], [
        39, 34449.5864302363], [
        39.5, 33454.1020252096], [
        40, 32378.2131052808], [
        40.5, 31183.7307867778], [
        41, 29838.0330597518], [
        41.5, 28385.3622429087], [
        42, 26827.2179208229], [
        42.5, 25148.6532523594], [
        43, 23363.4800821419], [
        43.5, 21486.2352554166], [
        44, 19515.5990578573], [
        44.5, 17485.8230224286], [
        45, 15426.5299774852], [
        45.5, 13358.604044273], [
        46, 11358.083722676], [
        46.5, 9436.51298627293], [
        47, 7630.34127604977], [
        47.5, 5990.38431855521], [
        48, 4539.644733189], [
        48.5, 3302.97929202387], [
        49, 2292.67594260067], [
        49.5, 1501.4545466168], [
        50, 923.694169087658], [
        50.5, 543.982082471506], [
        51, 354.154795938787], [
        51.5, 301.71406637277], [
        52, 282.301606719771], [
        52.5, 276.650423539557], [
        53, 283.770113971628], [
        53.5, 279.160099874783], [
        54, 284.964850865182], [
        54.5, 315.108419798419], [
        55, 362.343483288895], [
        55.5, 454.756304652686], [
        56, 601.826745717862], [
        56.5, 770.34219013351], [
        57, 969.193282691914], [
        57.5, 1175.78786628147], [
        58, 1389.45353171425], [
        58.5, 1602.84108681614], [
        59, 1812.21409604039], [
        59.5, 2011.3434625895], [
        60, 2196.25001786821], [
        60.5, 2366.86531721785], [
        61, 2523.79190450078], [
        61.5, 2659.64776347841], [
        62, 2777.19666818539], [
        62.5, 2873.06114820929], [
        63, 2957.03541018314], [
        63.5, 3024.96102634331], [
        64, 3072.00098671323], [
        64.5, 3109.85997696244], [
        65, 3136.8523231948], [
        65.5, 3149.18779369351], [
        66, 3151.40681507819], [
        66.5, 3144.49499665633], [
        67, 3125.57264344779], [
        67.5, 3100.26627976279], [
        68, 3071.22340710188], [
        68.5, 3033.45630282684], [
        69, 2988.89304594031], [
        69.5, 2944.73389007707], [
        70, 2893.28912295215], [
        70.5, 2841.92684660676], [
        71, 2785.73049744993], [
        71.5, 2727.8580276509], [
        72, 2674.15320283786], [
        72.5, 2609.66571079393], [
        73, 2550.68404299864], [
        73.5, 2485.88594263443], [
        74, 2425.00453450142], [
        74.5, 2361.77431962334], [
        75, 2304.62305017829], [
        75.5, 2240.02531052239], [
        76, 2182.59819335924], [
        76.5, 2118.94830363777], [
        77, 2062.40604479653], [
        77.5, 2001.10575522807], [
        78, 1945.4216999355], [
        78.5, 1892.70768669695], [
        79, 1833.50242596419], [
        79.5, 1784.5897701998], [
        80, 1733.33015648644], [
        80.5, 1681.46134548504], [
        81, 1629.90919884008], [
        81.5, 1581.3761785094], [
        82, 1536.64797802292], [
        82.5, 1491.76805117942], [
        83, 1448.25788683531], [
        83.5, 1404.74053203723], [
        84, 1362.31251078962], [
        84.5, 1322.68043611323], [
        85, 1281.30419501795], [
        85.5, 1245.11505219879], [
        86, 1209.22716170896], [
        86.5, 1168.84311676282], [
        87, 1137.64059452269], [
        87.5, 1104.72067202518], [
        88, 1069.37924172368], [
        88.5, 1037.81609534876], [
        89, 1009.70051246663], [
        89.5, 979.700402766309], [
        90, 950.616245511949], [
        90.5, 924.287383094869], [
        91, 897.386051643628], [
        91.5, 871.572470038367], [
        92, 846.489535008721], [
        92.5, 822.707699399251], [
        93, 801.604901855986], [
        93.5, 778.547722747723], [
        94, 751.64180197602], [
        94.5, 738.088281644767], [
        95, 715.588054378118], [
        95.5, 694.458547300904], [
        96, 678.61592681709], [
        96.5, 657.336870743172], [
        97, 643.683500488843], [
        97.5, 623.695184105331], [
        98, 609.638289681951], [
        98.5, 591.310961628716], [
        99, 578.005331622045], [
        99.5, 569.075815441019], [
        100, 687.282529011842]
    ])
    res_coil = get_res_coil(12.45)

    return res_amp, res_coil

def get_res_4():
    res_amp = np.array([[
        0.5, 749.353846153846], [
        1, 4870.800000000000], [
        2, 21799.384615384600], [
        3, 35424.000000000000], [
        4, 40873.846153846100], [
        5, 43598.769230769200], [
        6, 43598.769230769200], [
        7, 44280.000000000000], [
        8, 43939.384615384600], [
        9, 43258.153846153800], [
        10, 42236.307692307700], [
        11, 41214.461538461500], [
        12, 40192.615384615400], [
        13, 39170.769230769200], [
        14, 36786.461538461500], [
        15, 36105.230769230800], [
        16, 34061.538461538500], [
        17, 33039.692307692300], [
        18, 32017.846153846200], [
        19, 30655.384615384600], [
        20, 28271.076923076900], [
        21, 27589.846153846200], [
        22, 26057.076923076900], [
        23, 24524.307692307700], [
        24, 22991.538461538500], [
        25, 21629.076923076900], [
        26, 20436.923076923100], [
        27, 18904.153846153800], [
        28, 17371.384615384600], [
        29, 16179.230769230800], [
        30, 14816.769230769200], [
        31, 13624.615384615400], [
        32, 12687.923076923100], [
        33, 11580.923076923100], [
        34, 10473.923076923100], [
        35, 9281.769230769230], [
        36, 8311.015384615380], [
        37, 7425.415384615380], [
        38, 6539.815384615380], [
        39, 5858.584615384620], [
        40, 4836.738461538460], [
        41, 4189.569230769230], [
        42, 3542.400000000000], [
        43, 2997.415384615380], [
        44, 2384.307692307690], [
        45, 1873.384615384620], [
        46, 1362.461538461540], [
        47, 1021.846153846150], [
        48, 681.230769230769], [
        49, 408.738461538461], [
        50, 149.870769230769], [
        51, 102.184615384615], [
        52, 136.246153846154]
    ])
    res_amp[:, 1] = res_amp[:, 1] * new_logger_fact
    res_coil = get_res_coil(28.45)
    return res_amp, res_coil

def get_res_5():
    res_amp = np.array([[
        0.5, 1476], [
        1, 14040], [
        2, 70920], [
        3, 118800], [
        4, 151200], [
        5, 156600], [
        6, 162000], [
        7, 167400], [
        8, 171000], [
        9, 171000], [
        10, 169200], [
        11, 171000], [
        12, 169200], [
        13, 167400], [
        14, 167400], [
        15, 165600], [
        16, 160200], [
        17, 156600], [
        18, 153000], [
        19, 151920], [
        20, 147600], [
        21, 142560], [
        22, 136800], [
        23, 131040], [
        24, 125280], [
        25, 115200], [
        26, 108720], [
        27, 100800], [
        28, 93600], [
        29, 85680], [
        30, 77040], [
        31, 69120], [
        32, 61200], [
        33, 52560], [
        34, 45720], [
        35, 38520], [
        36, 32040], [
        37, 26280], [
        38, 21240], [
        39, 15480], [
        40, 12600], [
        41, 9720], [
        42, 6768], [
        43, 5040], [
        44, 3312], [
        45, 2232], [
        46, 1584], [
        47, 612], [
        48, 270], [
        49, 234], [
        50, 198], [
        51, 198], [
        52, 198]
    ])
    res_amp[:, 1] = res_amp[:, 1] * new_logger_fact
    res_coil = get_res_coil(70)
    return res_amp, res_coil

def lorentzian(f, *params):
    """
    Lorentzian model function used for fitting.
    """
    modes = len(params) // 3
    result = np.zeros_like(f)
    for i in range(modes):
        fc = params[i * 3]
        A = params[i * 3 + 1]
        Q = params[i * 3 + 2]
        result += A / (1 + 4 * Q ** 2 * ((f / fc) - 1) ** 2)
    result += params[-1]  # Background noise level (BN)
    return result

def gaussian_weights(f, mean, std, scale_factor=1.0):
    # Scale the width based on the data's frequency range (optional)
    std_adjusted = std * (np.max(f) - np.min(f)) / np.std(f)
    
    # Compute the Gaussian weights
    weights = np.exp(-((f - mean) ** 2) / (2 * std_adjusted ** 2))
    
    # Apply the scaling factor
    return scale_factor * weights

def sr_fit(f, p_in, modes):
    f_res = np.array([7.8, 14, 20, 27, 33, 39, 45])[:modes]
    ainits = [np.mean(p_in[(f > freq - 0.5) & (f < freq + 0.5)]) for freq in f_res]
    Qstart = 5
    init_params = []
    for i in range(modes):
        init_params.extend([f_res[i], ainits[i], Qstart])
    init_params.append(0)  # Background noise (BN) start value

    # Define bounds for parameters
    lower_bounds = []
    upper_bounds = []
    for i in range(modes):
        lower_bounds.extend([f_res[i] - 3, 0, 1])
        upper_bounds.extend([f_res[i] + 3, 2 * max(p_in), 20])
    lower_bounds.append(0)  # BN lower bound
    upper_bounds.append(max(p_in))  # BN upper bound

    means = np.mean(f)
    stds = np.std(f)
    weights = gaussian_weights(f, means, stds)

    # Fit the Lorentzian model to the data
    popt, _ = curve_fit(lorentzian, f, p_in*weights, p0=init_params, bounds=(lower_bounds, upper_bounds), sigma=weights)
    fitline = lorentzian(f, *popt)
    noiseline = popt[-1]
    results = np.reshape(popt[:-1], (modes, 3))

    return fitline, noiseline, results, None  # gof (goodness of fit) is not implemented


def srd_spec(t, fs, x, y):
    Frange = [3,48]

    Pov = 50
    Tseg = 20
    modes = 7

    NN = round(fs * Tseg)
    if NN >= len(x):
        NN = len(x)
    Nfft = NN
    if Nfft % 2 != 0:
        Nfft += 1
    w = windows.hamming(NN)

    # Welch's power spectral density estimate
    F, Pxx = welch(x, window=w, nperseg=NN, noverlap=int(NN * Pov / 100), nfft=Nfft, fs=fs)

    Pyy = None
    if len(y)>0:
        F2, Pyy = welch(y, window=w, nperseg=NN, noverlap=int(NN * Pov / 100), nfft=Nfft, fs=fs)

    ii = np.where((F >= Frange[0]) & (F <= Frange[1]))[0]
    F = F[ii]
    sreq1, sreq2 = get_equalizer(F, datetime.fromtimestamp(t))
    Pxx = Pxx[ii] * sreq1

    if len(y)>0:
        Pyy = Pyy[ii] * sreq2

    L1, noiseline1, R1, gof1 = sr_fit(F, Pxx, modes)
    L2, R2, gof2 = None, None, 0
    if len(y)>0:
        L2, noiseline2, R2, gof2 = sr_fit(F, Pyy, modes)

    return F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2


def print_srd_data(fn):
    """
    Prints the extracted SRD data in a meaningful way.

    Parameters:
    fn (str): The file name (path) of the SRD data file.
    """
    # Get metadata and data from the file
    downsampling_factor = 24

    start_reading = time.time()
    t, fs, x, y = read_srd_file(fn)
    reading_time = time.time() - start_reading
    print(f"Time to read srd: {reading_time:.4f} seconds")
    
    HNS_downsampled, HEW_downsampled = decimate_signal(x, y, downsampling_factor)
    decimated_frequency = fs / downsampling_factor
    # plot_PSD(f, S_NS, S_EW)
    print(f"Decimated frequency is: {decimated_frequency}")

    # np.savetxt('srd.txt', x, fmt='%d', newline='\n')
    print(len(x))
    print(len(y))
    print(len(HNS_downsampled))
    print(len(HEW_downsampled))

    with open('srd.txt', 'w') as f:
        for ns in HNS_downsampled:
            f.write(f"{ns:0.10f}\n")

    HNS_downsampled = np.loadtxt('srd.txt')

    plot_signal(HNS_downsampled, [], decimated_frequency, t)
    print(f"FS: {fs}")
    F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2 = srd_spec(t, fs, x, y)
    print(f"F {F.shape}, Pxx {Pxx.shape}")

    # Plotting the power spectral density (PSD)
    plt.figure()
    plt.plot(F, Pxx)
    plt.plot(F, L1, label='Lorentzian Fit')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [pT^2/Hz]')
    plt.title('Power Spectral Density (PSD) using Welch\'s Method')
    plt.grid()
    plt.xlim(0,50)
    plt.show()
    # S_NS, S_EW, f = compute_PSD(HNS_downsampled, [], decimated_frequency, 3, 48)
    # plot_PSD(f, S_NS, S_EW)

    # Save the data in .zst format to a buffer
    # buffer = io.BytesIO()
    # np.savez(buffer,
    #          NS=S_NS,
    #          # EW=HEW_downsampled
    #          )

    # Compress the buffer using zstandard
    # compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

    # Write the compressed data to a file
    # with open('srd_psd_downsampled.zst', 'wb') as f:
    #     f.write(compressed_data)

    # Print the extracted information in a meaningful way
    print(f"File: {fn}")
    print(f"Timestamp: {datetime.fromtimestamp(t)}")
    print(f"Sampling Frequency (fs): {fs} Hz")
    print(f"Channel Count: {'Single (x)' if len(y) == 0 else 'Dual (x and y)'}")
    print(f"First 10 Samples (Channel x): {x[:10]}")

    if len(y) > 0:
        print(f"First 10 Samples (Channel y): {y[:10]}")
    print(f"Number of Samples in x: {len(x)}")
    print(f"{len(x) / fs:.2f} seconds saved in file")
    if len(y) > 0:
        print(f"Number of Samples in y: {len(y)}")

def select_and_print_srd_data():
    """
    Opens a file dialog to select an SRD file and prints its data.
    """
    # Initialize the Tkinter root and hide the main window
    root = Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select the SRD file
    file_path = filedialog.askopenfilename(
        initialdir="/mnt/f/SouthStationSimple/Parnon260621", 
        title="Select an SRD File",
        filetypes=[("SRD Files", "*.SRD"), ("All Files", "*.*")]
    )

    # Check if a file was selected
    if file_path:
        print_srd_data(file_path)
    else:
        print("No file selected.")

# Run the file selection and data print function
select_and_print_srd_data()
