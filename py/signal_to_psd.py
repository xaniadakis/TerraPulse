import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import zstandard as zstd
import io
from tqdm import tqdm  # Import tqdm for the progress bar
import argparse
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import traceback
import datetime
from scipy.optimize import leastsq
import numpy as np
import pandas as pd
from colorama import Fore, Style
import os
import pandas as pd
from pathlib import Path
import mplcursors  # Add this library for interactivity

import logging


NUM_HARMONICS = 7  # Define the number of harmonics expected
FMIN = 3
FMAX = 48
DOWNSAMLPING_FACTOR = 30
SAMPLING_RATE = 5e6 / 128 / 13 / DOWNSAMLPING_FACTOR
INPUT_DIRECTORY = ''
FILE_TYPE = ''
WORKED = 0
new_logger_fact = 0.25  # New logger amp factor
old_logger = 0  # Global variable in the original code; define as needed
na_fits_num = 0
# Global list to store filenames of files with errors
error_files = []
fit_data = {'timestamp': [], 'NS_fit': [], 'EW_fit': []}
zero_fit_count = {'NS': 0, 'EW': 0}  # Counter for zero fits
output_means_file = INPUT_DIRECTORY+"/mean_fit_values.txt"  # Specify the output file path

def get_equalizer(freqs, date):
    # Define important dates
    date1 = datetime.datetime.strptime('12-Feb-2017', '%d-%b-%Y')
    date2 = datetime.datetime.strptime('10-Aug-2017', '%d-%b-%Y')
    date3 = datetime.datetime.strptime('03-Dec-2018', '%d-%b-%Y')
    date4 = datetime.datetime.strptime('01-Jul-2019', '%d-%b-%Y')

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

def get_res_coil(s):
    # f = freq vector, s = sensitivity (nVolt/pTesla/Hz)
    f = np.arange(1, 101).reshape(-1, 1)
    m = f * s
    out = np.hstack((f, m))
    return out

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

def plot_signal_on_fit_error(f, p_in, signal_filename, component, message=None):
    """
    Plot the signal and optionally display a message if fit fails.
    """
    plt.figure(figsize=(10, 6))
    if component == "NS":
        plt.plot(f, p_in, 'r', lw=1, label='$B_{NS}$ PSD')
    else:
        plt.plot(f, p_in, 'b', lw=1, label='$B_{EW}$ PSD')
    plt.title(f"{signal_filename} Frequency-Domain Signal\n")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(ls='--')
    plt.legend()
    if message:
        plt.annotate(message, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, color='red', ha='left', va='top')
    plt.show()

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
    # modes = len(params) // 3
    # result = np.zeros_like(f)
    # for i in range(modes):
    #     fc = params[i * 3]  # Center frequency
    #     A = params[i * 3 + 1]  # Amplitude
    #     Q = params[i * 3 + 2]  # Q factor

    #     # Lorentzian model
    #     lorentzian_part = A / (1 + 4 * Q**2 * ((f / fc) - 1)**2)

    #     # Gaussian model for broader contributions
    #     gaussian_part = A * np.exp(-((f - fc)**2) / (2 * (fc / 4)**2))

    #     # Blend the Lorentzian and Gaussian parts (adjust blend_factor as needed)
    #     blend_factor = 0.5
    #     result += (1 - blend_factor) * lorentzian_part + blend_factor * gaussian_part

    # # Background noise
    # result += params[-1]
    # return result


def gaussian_weights(f, mean, std, scale_factor=1.0):
    # Scale the width based on the data's frequency range (optional)
    std_adjusted = std * (np.max(f) - np.min(f)) / np.std(f)
    # Compute the Gaussian weights
    weights = np.exp(-((f - mean) ** 2) / (2 * std_adjusted ** 2))
    # Apply the scaling factor
    return scale_factor * weights

def residuals(params, f, p_in, func, weights):
    """
    Calculate the weighted residuals for leastsq.
    """
    return (func(f, *params) - p_in) * weights

def chi_squared(y_obs, y_model, weights):
    """
    Calculate Chi-squared statistic.
    """
    return np.sum(((y_obs - y_model) ** 2) / weights**2)

def r_squared(y_obs, y_model):
    """
    Calculate R-squared.
    """
    ss_res = np.sum((y_obs - y_model) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - (ss_res / ss_tot)

def rate_fit(chi2, r2, n, p, chi2_weight=0.5, r2_weight=0.5):
    """
    Rate the fit from 0 to 100 based on Chi-squared and R-squared.
    
    Args:
    chi2 (float): The chi-squared statistic.
    r2 (float): The R-squared value.
    n (int): The number of data points.
    p (int): The number of parameters in the model.
    chi2_weight (float): The weight for the Chi-squared metric (between 0 and 1).
    r2_weight (float): The weight for the R-squared metric (between 0 and 1).
    
    Returns:
    float: The fit rating between 0 and 100.
    """
    
    # Degrees of freedom
    dof = n - p
    
    # Normalize Chi-squared: We want a value near 1 for a good fit, so we scale it
    # Higher values of chi2 are worse, so we inverse the scaling (1 / (chi2 + 1)) to make it range between 0 and 1.
    chi2_normalized = 1 / (1 + chi2 / dof)
    
    # R-squared is already between 0 and 1, so we use it as is.
    r2_normalized = r2  # Since R^2 is between 0 and 1, we can use it directly.
    
    # Combine the two metrics to get a final score between 0 and 1
    final_score = (chi2_weight * chi2_normalized) + (r2_weight * r2_normalized)
    
    # Convert the score to a scale from 0 to 100
    fit_rating = final_score * 100
    
    return fit_rating


import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from scipy.signal import find_peaks, peak_prominences

# def dynamic_peak_detection(f, p_in, modes, prominence_factor=0.1):
#     # Calculate dynamic prominence based on the range of signal values
#     prominence = (np.max(p_in) - np.min(p_in)) * prominence_factor
    
#     # Detect peaks with dynamic prominence and minimum distance
#     peaks, _ = find_peaks(p_in, prominence=prominence)
    
#     # Select the top 'modes' peaks by prominence
#     if len(peaks) > modes:
#         prominences = peak_prominences(p_in, peaks)[0]
#         top_peaks = peaks[np.argsort(prominences)[-modes:]]
#         f_res = f[top_peaks]
#     else:
#         f_res = f[peaks] if peaks.size > 0 else np.linspace(np.min(f), np.max(f), modes)
#     return f_res

# def dynamic_weights(f, p_in, smoothing_factor=0.5):
#     # Smooth the signal to identify regions with high variability
#     smooth_signal = np.convolve(p_in, np.ones(5)/5, mode='same')
#     signal_variation = np.abs(p_in - smooth_signal)
    
#     # Assign higher weights to less variable regions
#     noise_level = np.std(p_in[:int(len(p_in) * 0.1)])  # Estimate from first 10% of data
#     weights = 1.0 / (1 + smoothing_factor * signal_variation / (noise_level + 1e-6))
#     return weights

# def rate_fit_dynamic(chi2, r2, n, p, signal_variance, chi2_weight=0.6, r2_weight=0.4):
#     dof = n - p
#     chi2_normalized = 1 / (1 + chi2 / (dof * signal_variance))
#     r2_normalized = np.clip(r2, 0, 1)
#     final_score = (chi2_weight * chi2_normalized) + (r2_weight * r2_normalized)
#     return final_score * 100

def adaptive_peak_detection(p_in, f, min_prominence=0.05, min_distance=2.0, max_peaks=None, smoothen=False):

    if smoothen:
        p_in = np.convolve(p_in, np.ones(5)/5, mode='same')

    # Detect peaks
    peaks, _ = find_peaks(p_in, prominence=(np.max(p_in) - np.min(p_in)) * min_prominence)
    # peaks, _ = find_peaks(p_in, prominence=(np.max(p_in) - np.min(p_in)) * min_prominence)

    if len(peaks) == 0:
        return np.array([]), []

    prominences = peak_prominences(p_in, peaks)[0]

    # Sort by prominence and keep only top peaks
    sorted_peaks = peaks[np.argsort(prominences)[::-1]]
    final_peaks = [sorted_peaks[0]]
    
    for peak in sorted_peaks[1:]:
        if np.all(np.abs(f[peak] - f[final_peaks]) > min_distance):
            final_peaks.append(peak)
        if max_peaks and len(final_peaks) >= max_peaks:
            break

    return f[final_peaks], final_peaks


def sr_fit(f, p_in, modes, signal_filename, component, smoothen=False):
    global na_fits_num

    # min_distance_hz = 3.3  # Minimum separation in Hz
    # df = f[1] - f[0]  # Frequency step size
    # distance = int(min_distance_hz / df)

    # # Initial frequency guesses using peak detection with a minimum distance
    # # peak_threshold = np.median(p_in) + 0.1 * (np.max(p_in) - np.median(p_in))
    # # peaks, _ = find_peaks(p_in, height=peak_threshold, distance=distance)
    # peaks, _ = find_peaks(p_in, height=np.max(p_in) * 0.1, distance=distance)
    # print(f"{component} peaks: {f[peaks]}")
    # if len(peaks) >= modes:
    #     f_res = f[peaks[:modes]]
    # else:
    #     # If fewer peaks than modes, spread frequencies evenly
    #     print(f"Warning: Detected fewer peaks ({len(peaks)}) than modes for {component}. Using default spacing.")
    #     f_res = np.linspace(np.min(f), np.max(f), modes)
    #     # f_res = f[peaks]

    # Schumann resonance harmonics (in Hz)
    # schumann_harmonics = np.array([7.8, 14, 20, 27, 33, 39, 45])  # Add or modify harmonics as needed
    # Peak detection
    # min_distance_hz = 4
    # df = f[1] - f[0]
    # distance = int(min_distance_hz / df)
    # peaks, _ = find_peaks(p_in, height=np.max(p_in) * 0.1, distance=distance)

    # Usage example:
    min_distance_hz = 2  # Adjust this to control peak merging sensitivity
    if np.all(p_in == 0) or len(p_in) == 0:
        logger.info(f"Skipping {signal_filename} - No significant signal detected for {component}")
        return None, None, None, 0.0

    f_res, peaks_indices = adaptive_peak_detection(p_in, f, min_prominence=0.05, 
                                                   min_distance=min_distance_hz, max_peaks=modes, smoothen=smoothen)
    # print(sorted(f_res))
    # if len(peaks) >= modes:
    #     # Use the detected peaks if sufficient
    #     f_res = f[peaks[:modes]]
    # else:
    #     f_res = schumann_harmonics[:modes]

    # Initial amplitude guesses around the detected peaks
    ainits = [np.mean(p_in[(f > freq - 0.5) & (f < freq + 0.5)]) for freq in f_res]
    # ainits = []
    # for freq in f_res:
    #     # Consider wider windows and penalize extreme variances
    #     local_region = p_in[(f > freq - 1) & (f < freq + 1)]
    #     if len(local_region) > 0:
    #         ainits.append(np.max(local_region) * 0.8)  # Start near local maximum amplitude
    #     else:
    #         ainits.append(np.mean(p_in) * 0.5)  # Fallback for sparse data regions

    # Set initial Q factors and background noise level
    # Qstart = 5
    Qstart = np.clip(10 / (np.max(f) - np.min(f)), 2, 50)  # Dynamic initial Q factor
    init_params = []
    for i in range(len(f_res)):
        init_params.extend([f_res[i], ainits[i], Qstart])
    init_params.append(0)  # Background noise (BN) start value

    # Define bounds for parameters
    # signal_range = np.max(p_in) - np.min(p_in)
    # peak_separation = np.abs(np.diff(f_res)).mean()  # Average peak separation

    lower_bounds = []
    upper_bounds = []
    for i in range(len(f_res)):
        # lower_bounds.extend([f_res[i] - 5, 0.5 * ainits[i], 1])
        # upper_bounds.extend([f_res[i] + 5, 1.5 * ainits[i], 50])
        # lower_bounds.extend([f_res[i] - 3, 0, 1])
        # upper_bounds.extend([f_res[i] + 3, 2 * max(p_in), 20])
        lower_bounds.extend([f_res[i] - 5, 0, 0.1])
        # upper_bounds.extend([f_res[i] + 5, 2 * max(p_in), 500])  # Allow high Q if needed
        upper_bounds.extend([f_res[i] + 5, 2 * max(p_in), 100])  # Allow wider Q ranges
        # lower_bounds.extend([f_res[i] - 5, 0, max(0.1, peak_separation / 10)])
        # upper_bounds.extend([f_res[i] + 5, 2 * max(p_in), 150])  # Wider Q range for complex modes

    lower_bounds.append(0)  # BN lower bound
    upper_bounds.append(max(p_in))  # BN upper bound

    # Compute Gaussian weights
    mean = np.mean(f)
    std = np.std(f)
    noise_level = np.std(p_in[:int(len(p_in) * 0.1)])  # Estimate noise from the first 10% of data
    weights = gaussian_weights(f, mean, std, scale_factor=1.0) / (noise_level + 1e-6)
    # weights = gaussian_weights(f, mean, std)
    # signal_variation = np.abs(p_in - np.convolve(p_in, np.ones(10)/10, mode='same'))
    # weights = 1.0 / (1 + 0.5 * signal_variation / (noise_level + 1e-6))

    try:
        # f_res = dynamic_peak_detection(f, p_in, modes)
        # weights = dynamic_weights(f, p_in)

        # result = least_squares(
        #     residuals,
        #     init_params,
        #     bounds=(lower_bounds, upper_bounds),
        #     args=(f, p_in, lorentzian, weights),
        #     method='trf',
        #     loss='huber',
        #     f_scale=0.5
        # )

        # if result.success:
        #     fitline = lorentzian(f, *result.x)
        #     chi2 = np.sum(((p_in - fitline) ** 2) / weights**2)
        #     r2 = r_squared(p_in, fitline)
        #     fit_rating = rate_fit_dynamic(chi2, r2, len(f), len(result.x), np.var(p_in))
        #     return fitline, result.x[-1], np.reshape(result.x[:-1], (modes, 3)), fit_rating
        # else:   
        #     raise RuntimeError(result.message)
        # Use least_squares for fitting
        result = least_squares(
            residuals,
            init_params,
            bounds=(lower_bounds, upper_bounds),
            args=(f, p_in, lorentzian, weights),
            method='trf',
            loss='soft_l1',  # Robust loss to handle outliers
            # max_nfev=10000
        )

        if not result.success:
            raise RuntimeError(result.message)

        params = result.x

        # Extract the fitted values
        fitline = lorentzian(f, *params)
        noiseline = params[-1]
        results = np.reshape(params[:-1], (len(f_res), 3))

        # Calculate fit rating (Chi-squared and R-squared metrics can be added if needed)
        chi2 = np.sum(((p_in - fitline) ** 2) / weights**2)
        r2 = 1 - np.sum((p_in - fitline) ** 2) / np.sum((p_in - np.mean(p_in)) ** 2)
        # aic = len(f) * np.log(np.sum((p_in - fitline) ** 2) / len(f)) + 2 * len(params)
        fit_rating = (0.5 * (1 / (1 + chi2 / (len(f) - len(params))))) + (0.5 * r2)
        # fit_rating = (0.333 * r2) + (0.333 * (1 / (1 + chi2))) + (0.333 * (1 / (1 + aic)))
        fit_rating *= 100
        # print(f"{component} chi2: {(1 / (1 + chi2 / (len(f) - len(params))))}, r2:{r2}")
        return fitline, noiseline, results, fit_rating

    except RuntimeError as e:
        na_fits_num += 1
        logger.info(f"{na_fits_num} N/A fit for {component} of {signal_filename} (conv err)")
        return None, None, None, 0.0

    except Exception as e:
        na_fits_num += 1
        logger.info(f"{na_fits_num} N/A fit for {component} of {signal_filename} (ex)")
        return None, None, None, 0.0
    
# def sr_fit(f, p_in, modes, signal_filename, component):
#     global na_fits_num
#     f_res = np.array([7.8, 14, 20, 27, 33, 39, 45])[:modes]
#     ainits = [np.mean(p_in[(f > freq - 0.5) & (f < freq + 0.5)]) for freq in f_res]
#     Qstart = 5
#     init_params = []
#     for i in range(modes):
#         init_params.extend([f_res[i], ainits[i], Qstart])
#     init_params.append(0)  # Background noise (BN) start value

#     # Define bounds for parameters
#     lower_bounds = []
#     upper_bounds = []
#     for i in range(modes):
#         lower_bounds.extend([f_res[i] - 3, 0, 1])
#         upper_bounds.extend([f_res[i] + 3, 2 * max(p_in), 20])
#     lower_bounds.append(0)  # BN lower bound
#     upper_bounds.append(max(p_in))  # BN upper bound

#     mean = np.mean(f)
#     std = np.std(f)
#     weights = gaussian_weights(f, mean, std)

#     try:
#         # Use leastsq for fitting
#         params, cov_x, infodict, mesg, ier = leastsq(
#             residuals, init_params, args=(f, p_in * weights, lorentzian, weights), full_output=True, maxfev=10000
#         )

#         if ier not in [1, 2, 3, 4]:
#             # Fit did not converge; ier values other than 1-4 indicate failure
#             raise RuntimeError(f"{mesg}")

#         # Extract the fitted values
#         fitline = lorentzian(f, *params)
#         _ = params[-1] # supposed to be the noiseline
#         results = np.reshape(params[:-1], (modes, 3))

#         # Calculate the fit rating from 0 to 100
#         fit_rating = rate_fit(chi_squared(p_in, fitline, weights), 
#                             r_squared(p_in, fitline), 
#                             len(f), 
#                             len(params), 
#                             chi2_weight=0.5, 
#                             r2_weight=0.5)
        
#         return fitline, _, results, fit_rating
#     except RuntimeError as e:
#         # Explicitly handle the fit failure
#         # print("Fit failed: No fit available for the current data.")
#         na_fits_num += 1
#         print(f"\r{na_fits_num} N/A fit for {component} of {signal_filename} -> Error: {str(e)}")
#         # plot_signal_on_fit_error(f, p_in, signal_filename, component, message=str(e))
#         return None, None, None, 0.0

#     except Exception as e:
#         # Handle any other unexpected errors
#         # print("An unexpected error occurred.")
#         na_fits_num += 1
#         print(f"\r{na_fits_num} N/A fit for {component} of {signal_filename} -> Error: {str(e)}")
#         # plot_signal_on_fit_error(f, p_in, signal_filename, component, message="Unexpected error during fit.")
#         return None, None, None, 0.0

def validate_file_type(file_type):
    if file_type not in ['.pol', '.hel']:
        raise ValueError("Invalid file type. Only '.pol' and '.hel' are supported.")
    return file_type

def get_file_size(file_path):
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb

def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        logger.error(f"The file does not exist: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while deleting the file: {e}")

def read_signals(file_path):
    with open(file_path, 'r') as file:
        hns_features = np.array([float(x) for x in file.readline().strip().split('\t')])
        hew_features = np.array([float(x) for x in file.readline().strip().split('\t')])
        harmonics = []
        for _ in range(NUM_HARMONICS):
            harmonics.append(np.array([float(x) for x in file.readline().strip().split('\t')]))
        data = np.loadtxt(file, delimiter='\t', dtype=int)
    return hns_features, hew_features, harmonics, data


def transform_signal(input_filename, file_extension, do_plot=False, do_not_fit=False, gui=False):
    global WORKED, fit_data
    try:
        # Load data from the file
        data = np.loadtxt(input_filename+file_extension, delimiter='\t')
        # Extract filename and parse date-time information
        base_filename = os.path.basename(input_filename)
        date_time_str = os.path.splitext(base_filename)[0]
        file_origin = "Hellenic" if file_extension == '.hel' else "Polski" if file_extension == '.pol' else "Unknown"

        try:
            if file_origin == "Hellenic":
                file_datetime = datetime.datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
                formatted_datetime = file_datetime.strftime("%Y-%m-%d %H:%M:%S")
            else:
                file_datetime = datetime.datetime.strptime(date_time_str, "%Y%m%d%H%M")
                formatted_datetime = file_datetime.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            formatted_datetime = "Unknown Date-Time"

        # Determine if the file has a single column or multiple columns
        if data.ndim == 1:  # Single-column data
            # print("Detected single-column data.")
            HNS = data  # Treat the single column as HNS
            HEW = None  # No second channel
        elif data.ndim == 2:  # Multi-column data
            # print("Detected multi-column data.")
            HNS = data[:, 0]
            HEW = data[:, 1] if data.shape[1] > 1 else None
        else:
            raise ValueError(f"Unexpected file format: data has invalid dimensions {data.ndim}.")

        # Time-domain plotting
        if do_plot:
            timespace = np.linspace(0, len(HNS) / SAMPLING_RATE, len(HNS))
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(timespace, HNS, 'r', lw=1, label=r'$B_{NS}$')
            if HEW is not None:
                plt.plot(timespace, HEW, 'b', lw=1, label=r'$B_{EW}$')
            plt.title(f"{file_origin}-Logger Time-Domain Signal\n{formatted_datetime}")
            plt.ylabel("B [pT]")
            plt.xlabel("Time [sec]")
            plt.grid(ls=':')
            plt.legend()

        M = int(20 * SAMPLING_RATE)
        overlap = M // 2
        w = signal.windows.hamming(M)
        sreq1, sreq2 = None, None
        # Welch's power spectral density estimate
        if len(HNS) < M:
            logger.info(f"Warning: NS Signal length {len(HNS)} is shorter than nperseg {M}.")    
            error_files.append(input_filename+": NS signal shorter than window") 
            return
        frequencies, S_NS = signal.welch(x=HNS, window=w, fs=SAMPLING_RATE, nperseg=M, noverlap=overlap, scaling='spectrum')

        # Create a frequency mask based on FMIN and FMAX
        mask = (frequencies > FMIN) & (frequencies < FMAX)
        # Apply the mask to frequencies and S_NS
        frequencies = frequencies[mask]
        S_NS = S_NS[mask]
        S_NS /= (frequencies[1] - frequencies[0])
        # Adjust S_NS depending on the file origin
        if file_origin == "Hellenic":
            sreq1, sreq2 = get_equalizer(frequencies, file_datetime)
            S_NS *= sreq1
        if do_not_fit:
            L1, R1, gof1 = None, None, None
        else:
            L1, _, R1, gof1 = sr_fit(frequencies, S_NS, NUM_HARMONICS, input_filename, "NS")
            if gof1 < 50:
                _L1, _, _R1, _gof1 = sr_fit(frequencies, S_NS, NUM_HARMONICS, input_filename, "NS", smoothen=True)
                if _gof1>gof1:
                    print(f"We got {_gof1} gof instead of {gof1} by smoothing NS.")
                    L1, R1, gof1 = _L1, _R1, _gof1
        L2, R2, gof2 = None, None, None
        if HEW is not None:
            # Compute PSD for HEW if it exists
            # frequencies, S_EW = signal.welch(HEW, fs=SAMPLING_RATE, nperseg=M, noverlap=overlap, scaling='spectrum')
            # if file_origin == "Hellenic":
            #     frequencies = frequencies[ii]
            #     S_EW = S_EW[ii] * sreq2
            # elif file_origin == "Polski":
            #     S_EW = S_EW / (frequencies[1] - frequencies[0])
            #     S_EW = S_EW[mask]
            #     frequencies = frequencies[mask]

            # Compute the Welch power spectral density estimate for the second signal
            if len(HEW) < M:
                logger.info(f"Warning: EW Signal length {len(HEW)} is shorter than nperseg {M}.")        
                error_files.append(input_filename+": EW signal shorter than window") 
                return
            frequencies, S_EW = signal.welch(HEW, fs=SAMPLING_RATE, nperseg=M, noverlap=overlap, scaling='spectrum')

            # Apply the same frequency mask based on FMIN and FMAX
            mask = (frequencies > FMIN) & (frequencies < FMAX)

            # Apply the mask to frequencies and S_EW
            frequencies = frequencies[mask]
            S_EW = S_EW[mask]
            S_EW /= (frequencies[1] - frequencies[0])

            # Adjust S_EW depending on the file origin
            if file_origin == "Hellenic":
                S_EW *= sreq2

            if not do_not_fit:
                L2, _, R2, gof2 = sr_fit(frequencies, S_EW, NUM_HARMONICS, input_filename, "EW")
                if gof2 < 50:
                    _L2, _, _R2, _gof2 = sr_fit(frequencies, S_EW, NUM_HARMONICS, input_filename, "EW", smoothen=True)
                    if _gof2>gof2:
                        print(f"We got {_gof2} gof instead of {gof2} by smoothing EW.")
                        L2, R2, gof2 = _L2, _R2, _gof2
        else:
            S_EW = None

        # Frequency-domain plotting
        if do_plot:
            plt.subplot(2, 1, 2)
            if L1 is not None and L1.any():
                plt.plot(frequencies, S_NS, 'r', lw=1, label='$B_{NS}$ PSD')
                plt.plot(frequencies, L1, label='$B_{NS}$ '+f'Lorentzian Fit: {gof1:.2f}')  
            else:
                plt.plot(frequencies, S_NS, 'r', lw=1, label='$B_{NS}$ PSD (Fit N/A)')
            # plt.plot(frequencies, noiseline1 * np.ones_like(frequencies), '--', label='Noise Line')
            # plt.annotate(f"Fit Rating: {gof1:.2f}", xy=(0.65, 0.85), xycoords='axes fraction', fontsize=10, color='red')
            if S_EW is not None:
                if L2 is not None and L2.any():
                    plt.plot(frequencies, S_EW, 'b', lw=1, label='$B_{EW}$ PSD')
                    plt.plot(frequencies, L2, label='$B_{EW}$ '+f'EW Lorentzian Fit: {gof2:.2f}')
                else:
                    plt.plot(frequencies, S_EW, 'b', lw=1, label='$B_{EW}$ PSD (Fit N/A)')
                # plt.plot(frequencies, noiseline2 * np.ones_like(frequencies), '--', label='Noise Line')
                # plt.annotate(f"Fit Rating: {gof2:.2f}", xy=(0.65, 0.75), xycoords='axes fraction', fontsize=10, color='blue')
            plt.title(f"{file_origin}-Logger Frequency-Domain Signal\n{formatted_datetime}")
            plt.ylabel(r"$PSD\ [pT^2/Hz]$")
            plt.xlabel("Frequency [Hz]")
            plt.xticks(np.arange(0, 50, step=5))
            plt.grid(ls=':')
            plt.legend()
            
            # Add mplcursors for interactivity
            cursor = mplcursors.cursor(hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(f"x={sel.target[0]:.2f}, y={sel.target[1]:.2f}"))

            plt.tight_layout()
            if not gui:
                plt.show()
            else:
                return plt.gcf()

        # Save results
        buffer = io.BytesIO()
        np.savez(buffer, 
                freqs=frequencies, 
                NS=S_NS, 
                EW=S_EW if S_EW is not None else np.array([]), 
                gof1=gof1, 
                gof2=gof2, 
                **({"R1": R1} if R1 is not None else {}), 
                **({"R2": R2} if R2 is not None else {}))
        compressed_data = zstd.ZstdCompressor(level=3).compress(buffer.getvalue())

        with open(os.path.splitext(input_filename)[0] + '.zst', 'wb') as f:
            f.write(compressed_data)
        WORKED += 1

        fit_data['timestamp'].append(file_datetime)
        if not do_not_fit:
            fit_data['NS_fit'].append(gof1 if gof1 > 0 else np.nan)
            fit_data['EW_fit'].append(gof2 if HEW is not None and gof2 > 0 else (np.nan if HEW is not None else None))

    except IndexError as ie:
        logger.info(f"Indexing error occurred while processing '{input_filename+file_extension}': {repr(ie)}")
        logger.info(f"Shape of data at error: {data.shape}")
        traceback.print_exc() 
        error_files.append(input_filename)    
    except ValueError as ve:
        logger.info(f"Value error occurred while processing '{input_filename+file_extension}': {repr(ve)[:500]}")
        traceback.print_exc() 
        error_files.append(input_filename)
    except Exception as e:
        logger.info(f"An unexpected error occurred while processing '{input_filename+file_extension}': {repr(e)}")
        traceback.print_exc() 
        error_files.append(input_filename)


def process_files_in_directory():
    global na_fits_num

    # Pre-index all `.zst` files for fast lookup
    processed_files = set()
    for root, _, files in os.walk(INPUT_DIRECTORY):
        processed_files.update(
            os.path.join(root, f[:-4]) for f in files if f.endswith('.zst')
        )

    # Gather files from each subdirectory and process them as packets
    subdirectory_files = {}  # Dictionary to store files grouped by subdirectories
    for root, dirs, files in os.walk(INPUT_DIRECTORY):
        signal_files = [f for f in files if f.endswith(FILE_TYPE)]
        if signal_files:
            subdirectory_files[root] = [
                os.path.join(root, os.path.splitext(f)[0]) for f in signal_files
            ]

    if not subdirectory_files:
        logger.info(f"{Fore.RED}No {FILE_TYPE} files found in {INPUT_DIRECTORY} directory!{Style.RESET_ALL}")
        return

    logger.info(f"Will transform files grouped in {len(subdirectory_files)} subdirectories!\n")

    # Create an outer progress bar for subdirectory processing
    subdir_pbar = tqdm(
        total=len(subdirectory_files), 
        desc=f"{Fore.CYAN}Total{Style.RESET_ALL}", 
        unit="d", 
        leave=True, 
        bar_format="{l_bar}\033[44m{bar}\033[0m| {n_fmt}/{total_fmt} {unit} | el: {elapsed} | rem: {remaining}",
        ncols=100  
    )

    for i, (subdir, file_list) in enumerate(subdirectory_files.items()):
        unprocessed_files = [f for f in file_list if f not in processed_files]
        if not unprocessed_files:
            logger.info(f"Skipping subdirectory {i}: {subdir} (All files already processed)")
            continue  # Skip to the next subdirectory

        # Prepare the directory information
        # tqdm.write(f"\nProcessing subdirectory {i + 1}/{len(subdirectory_files)}: {subdir} with {len(file_list)} files")

        # Inner progress bar for files in the current subdirectory
        # file_pbar = tqdm(
        #     total=len(file_list), 
        #     desc=f"{Fore.YELLOW}..../{Path(subdir).name}/{Style.RESET_ALL}", 
        #     unit="files", 
        #     leave=False, 
        #     bar_format="{l_bar}\033[44m{bar}\033[0m| {n_fmt}/{total_fmt} {unit} | Elapsed: {elapsed}",
        #     ncols=150  
        # )
        for input_filename in file_list:
            transform_signal(input_filename, FILE_TYPE, do_plot=False)
            # file_pbar.update(1)  # Update file progress bar
        # file_pbar.close()  # Close the inner progress bar once done

        # If valid fit data exists, process and print statistics
        if fit_data['timestamp']:
            df = pd.DataFrame(fit_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df['NS_zero_fit'] = df['NS_fit'].apply(lambda x: 1 if pd.isna(x) else 0)
            df['EW_zero_fit'] = df['EW_fit'].apply(lambda x: 1 if pd.isna(x) else 0)

            df.set_index('timestamp', inplace=True)
            # rolling_stats = df.resample('24h').agg({
            #     'NS_fit': ['mean', 'std', 'count'],
            #     'EW_fit': ['mean', 'std', 'count'],
            #     'NS_zero_fit': 'sum',
            #     'EW_zero_fit': 'sum'
            # })

            # if not rolling_stats.empty:
            #     latest_stats = rolling_stats.iloc[-1]
            #     window_start = rolling_stats.index[-1]
            #     window_end = window_start + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)

            #     if window_start.date() == window_end.date():
            #         tqdm.write(f"Date: {window_start.strftime('%Y-%m-%d')}")
            #     else:
            #         tqdm.write(f"Datetime Window From: {window_start.strftime('%Y-%m-%d %H:%M:%S')} To: {window_end.strftime('%Y-%m-%d %H:%M:%S')}")
            #     tqdm.write(f"  NS Mean: {latest_stats[('NS_fit', 'mean')]:.2f}, NS Std: {latest_stats[('NS_fit', 'std')]:.2f}, NS Count: {int(latest_stats[('NS_fit', 'count')])}")
            #     tqdm.write(f"  EW Mean: {latest_stats[('EW_fit', 'mean')]:.2f}, EW Std: {latest_stats[('EW_fit', 'std')]:.2f}, EW Count: {int(latest_stats[('EW_fit', 'count')])}")
            #     tqdm.write("-" * 40)
            # na_fits_num += df['EW_zero_fit'].sum() + df['NS_zero_fit'].sum()
        else:
            logger.info(f"{Fore.RED}No valid fit data collected.{Style.RESET_ALL}")

        subdir_pbar.update(1)

    subdir_pbar.close()  # Close the outer progress bar once done

import tkinter as tk
from tkinter import filedialog

def select_file_and_transform(file_path=None, do_not_fit=False):
    if file_path is None:

        # Initialize Tkinter file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open file dialog to select a single file
        file_path = filedialog.askopenfilename(
            # initialdir=INPUT_DIRECTORY, 
            title="Select a signal file",
            filetypes=[("POL Files", "*.pol"), ("HEL Files", "*.hel"), ("All Files", "*.*")]
        )

    # If a file was selected, process it
    if file_path:
        base_filepath, file_extension = os.path.splitext(file_path)
        if file_extension != ".hel" and file_extension != ".pol":
            logger.info("Only handles [.hel/.pol files], please try again!")
            exit(1)
        transform_signal(base_filepath, file_extension, do_plot=True, do_not_fit=do_not_fit)  # Enable plotting
    else:
        logger.info("No file selected.")

def translate_windows_to_linux_path(windows_path):
    """
    Converts a Windows file path to a Linux file path, handling whitespaces.
    
    Args:
        windows_path (str): The Windows file path to convert.
    
    Returns:
        str: The converted Linux file path.
    """
    windows_path = windows_path.strip()
    linux_path = windows_path.replace("\\", "/")
    if ":" in linux_path:
        drive, path = linux_path.split(":", 1)
        linux_path = f"/mnt/{drive.lower()}{path}"
    return linux_path

def start_gui_browser(file_extension=".pol", do_not_fit=True):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import os
    import datetime

    current_index = 0
    file_list = []
    canvas = None
    toolbar = None
    current_folder = ""

    root = tk.Tk()
    root.title("Signal Explorer")

    control_frame = tk.Frame(root)
    control_frame.pack(padx=10, pady=10)

    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    def load_files_from_folder(folder):
        nonlocal file_list, current_index, current_folder
        file_list = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(file_extension)
        ])
        if not file_list:
            messagebox.showwarning("No Files", f"No {file_extension} files found.")
            return
        current_index = 0
        current_folder = folder
        for widget in goto_widgets:
            widget.grid()
        load_signal()

    def browse_folder():
        folder_selected = filedialog.askdirectory(title="Select Signal Folder")
        if not folder_selected:
            return
        load_files_from_folder(folder_selected)

    def load_signal():
        nonlocal canvas, toolbar
        if not file_list:
            return
        file_path = file_list[current_index]
        base_path, ext = os.path.splitext(file_path)
        print(f"Loading: {file_path}")
        fig = transform_signal(base_path, ext, do_plot=True, do_not_fit=do_not_fit, gui=True)
        if fig is None:
            return
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def next_signal():
        nonlocal current_index
        if current_index < len(file_list) - 1:
            current_index += 1
            load_signal()

    def prev_signal():
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            load_signal()

    def goto_time():
        nonlocal current_index
        try:
            hh = int(hour_entry.get())
            mm = int(minute_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Hour and minute must be integers.")
            return
        if not (0 <= hh <= 23):
            messagebox.showerror("Invalid Hour", "Hour must be between 0 and 23.")
            return
        if not (0 <= mm <= 59):
            messagebox.showerror("Invalid Minute", "Minute must be between 0 and 59.")
            return
        mm -= mm % 5
        hh_str = f"{hh:02d}"
        mm_str = f"{mm:02d}"
        for i, f in enumerate(file_list):
            fname = os.path.basename(f)
            if file_extension == ".hel" and fname[8:12] == hh_str + mm_str:
                current_index = i
                load_signal()
                return
            elif file_extension == ".pol" and fname[8:10] == hh_str and fname[10:12] == mm_str:
                current_index = i
                load_signal()
                return
        messagebox.showinfo("Not Found", f"No file found with time {hh_str}:{mm_str}")

    def load_from_calendar():
        nonlocal current_folder
        y = year_var.get()
        m = month_var.get()
        d = day_var.get()
        if not (y and m and d):
            messagebox.showerror("Invalid Date", "Please select year, month, and day.")
            return
        try:
            selected_date = datetime.date(int(y), int(m), int(d))
            parent_dir = os.path.dirname(current_folder)
            target_folder = os.path.join(parent_dir, selected_date.strftime("%Y%m%d"))  # fixed here
            if os.path.isdir(target_folder):
                load_files_from_folder(target_folder)
            else:
                messagebox.showinfo("Not Found", f"No folder for {selected_date.strftime('%Y%m%d')}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Main controls
    tk.Button(control_frame, text="Select Folder", command=browse_folder).grid(row=0, column=0, padx=5)
    tk.Button(control_frame, text="Previous", command=prev_signal).grid(row=0, column=1, padx=5)
    tk.Button(control_frame, text="Next", command=next_signal).grid(row=0, column=2, padx=5)

    hour_label = tk.Label(control_frame, text="Hour:")
    hour_entry = tk.Entry(control_frame, width=3)
    minute_label = tk.Label(control_frame, text="Minute:")
    minute_entry = tk.Entry(control_frame, width=3)
    goto_button = tk.Button(control_frame, text="Go To", command=goto_time)

    goto_widgets = [hour_label, hour_entry, minute_label, minute_entry, goto_button]
    positions = [3, 4, 5, 6, 7]
    for widget, col in zip(goto_widgets, positions):
        widget.grid(row=0, column=col, padx=(5 if col in (3, 5) else 2), sticky="w")
        widget.grid_remove()

    # Calendar dropdowns
    year_var = tk.StringVar()
    month_var = tk.StringVar()
    day_var = tk.StringVar()
    current_year = datetime.datetime.now().year
    years = [str(y) for y in range(2015, current_year + 1)]
    months = [str(m).zfill(2) for m in range(1, 13)]
    days = [str(d).zfill(2) for d in range(1, 32)]

    tk.Label(control_frame).grid(row=1, column=0, pady=5)
    tk.OptionMenu(control_frame, year_var, *years).grid(row=1, column=1)
    tk.OptionMenu(control_frame, month_var, *months).grid(row=1, column=2)
    tk.OptionMenu(control_frame, day_var, *days).grid(row=1, column=3)
    tk.Button(control_frame, text="Load Date Folder", command=load_from_calendar).grid(row=1, column=4, padx=5)

    root.mainloop()

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Time-Domain Signal To Power Spectral Density")
    parser.add_argument(
        "--file-select", 
        action="store_true", 
        help="Enable file selection mode to process a single file using a file dialog"
    )
    parser.add_argument(
        "--file-path", 
        type=str, 
        help="Specify the file path directly instead of selecting from the dialog"
    )
    parser.add_argument(
        "-t", "--file-type", 
        choices=['pol', 'hel'], 
        help="Specify the file type to process. Only 'pol' or 'hel' are allowed."
    )
    parser.add_argument(
        "-d", "--input-directory", 
        default="../output/", 
        help="Specify the input directory containing files to process. Default is '../output/'."
    )
    parser.add_argument(
        "-l", "--log-file", 
        default=None,
        help="Specify the file to log at."
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Disable Lorentzian fitting and only compute PSD"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI mode to browse and plot signals in a folder"
    )
    args = parser.parse_args()


    # Setup logging
    if args.log_file != None:
        log_file_path = args.log_file
    else: 
        log_file_path = "./temp.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()  # Also print logs to the console
        ]
    )
    logger = logging.getLogger(__name__)

    # Adjust `file-type` requirement based on `file-select`
    if not (args.file_select or args.file_path) and args.file_type is None:
        parser.error("-t/--file-type argument is required")

    if args.gui:
        start_gui_browser(file_extension=f".{args.file_type or 'pol'}", do_not_fit=args.no_fit)
        exit(0)

    # Configuration
    INPUT_DIRECTORY = args.input_directory # Root directory containing all date subdirectories

    if args.file_path:
        # Direct file path mode
        select_file_and_transform(translate_windows_to_linux_path(args.file_path), args.no_fit)
        exit(0)
    elif args.file_select:
        # File selection mode
        print(f"Please select the singal you want to plot.")
        select_file_and_transform()
        exit(0)
    else:
        # Set global FILE_TYPE based on user input
        FILE_TYPE = f".{args.file_type}"
        validate_file_type(FILE_TYPE)  # Validate the file type
        logger.info(f"Will process {FILE_TYPE} files from {INPUT_DIRECTORY} dir.")
        # Directory processing mode
        start_time = time.time()

        process_files_in_directory()
        save_zstd_time = time.time() - start_time
        if WORKED>0:
            logger.info(f"Converting to zstd file format took: {save_zstd_time} seconds")
             # Write error log to file if there are any errors
            if error_files:
                with open("error_log.txt", "w") as log_file:
                    log_file.write("Files with errors:\n")
                    log_file.write("\n".join(error_files))
                logger.error(f"Error log written to 'error_log.txt' with {len(error_files)} entries. (transformed {WORKED} files, n/a fit for {na_fits_num} signals)")
            else:
                logger.info(f"No errors encountered. (transformed {WORKED} files, n/a fit for {na_fits_num} signals)")
            exit(0)
        else:
            logger.info(f"Nothing happened. (transformed {WORKED} files, n/a fit for {na_fits_num} signals)")
            exit(1)