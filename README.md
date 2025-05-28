
compile and run signal to zst flow by running the bash script:
 > ./signalforge.sh pol /mnt/f/POLISH\ DATA/Raw\ Data/JanuszNew /mnt/e/POLSKI_DB
 > ./signalforge.sh hel /mnt/e/KalpakiSortedData/180102_15 /mnt/e/HELLENIC_DB
 > python3 signalforge.py hel /mnt/e/KalpakiSortedData/180102_15 /mnt/e/HELLENIC_DB

if you need to plot psd from c (not properly working yet):
 > gnuplot -e "set terminal png; set output '202301060125_psd.png'; set xlabel 'Frequency (Hz)'; set ylabel 'PSD'; set xrange [0:50]; set yrange [0:0.6]; plot './output/202301060125_psd.txt' using 1:2 with lines title 'PSD'"

To convert my code from matlab to C, I run this:
 > codegen lorentzian_fit_psd -args {coder.typeof(0, [10000, 1], [1, 0]), coder.typeof(0, [10000, 1], [1, 0]), 0} -o lorentzian_fit_psd_codegen -config:lib -g

we need to include those headers to potentially run the code:
 > gcc -I/usr/local/MATLAB/R2024a/extern/include -I/home/vag/PycharmProjects/TerraPulse/matlab/codegen/lib/lorentzian_fit_psd main.c -o main


python3 signalforge.py hel 'F:\SouthStationSimple' 'E:\HELLENIC_DB'
run hel /mnt/f/SouthStationSimple /mnt/f/HELLENIC_DB


mount external hdd to wsl:
sudo mkdir -p /mnt/g
sudo mount -t drvfs G: /mnt/g
sudo umount /mnt/f

udisksctl unmount -b /mnt/f
udisksctl power-off -b /mnt/f


python3 signalforge.py hel 'F:\Βόρειος Σταθμός Simple' 'F:\ΒΟΡΕΙΟΣ Σταθμός (506gb)' 'E:\NEW_NORTH_HELLENIC_DB' 

python3 signalforge.py pol 'E:\MEGA_POLISH_DATA\Summer 24\20240620' '../testthisone/'

python3 signalforge.py pol 'E:\MEGA_POLISH_DATA' 'F:\TRIPLE and DOUBLE\ΠΟΛΟΝΙΚΟ ΣΥΣΤΗΜΑ' 'F:\POLISH DATA\Raw Data' 'F:\ΝΟΤΕΙΟΣ Σταθμός (867Gb)\RAW DATA POLISH' 'E:\POLISH_DATA\Raw_Data' 'E:\POLSKI_DB'

python3 py/plot_period_spectograms.py -d '/mnt/e/AGAIN_NORTH_HELLENIC_DB' -o '../testspechelnORTH' -t hel -y 2020
python3 py/plot_period_spectograms.py -d '/mnt/e/POLSKI_DB' -o '../testspecpol' -t pol -y 2024

python3 py/plot_period_spectograms.py -d '/mnt/e/NEW_POLSKI_DB' -o '../spectrograms_pol' -t pol

python3 signalforge.py hel '\mnt\g\Greek Data\Kalpaki' '\mnt\g\Greek Data\Kalpaki\SRdatataxinomimena Triple' '\mnt\g\Greek Data\Kalpaki\Tests north' '\mnt\g\Greek Data\Kalpaki\Tests north\NEOCHORI' '\mnt\g\Kalpaki20170104' '\mnt\g\Kalpaki4' '\mnt\g\Kalpaki5' '\mnt\g\SRD Sata\ΜΕΤΑΦΟΡΑ\DATAnew' '\mnt\g\SRD Sata\ΜΕΤΑΦΟΡΑ\DATAnew\KALNS190116till220416' '\mnt\g\SRD Sata\ΜΕΤΑΦΟΡΑ\DATAnew\KalNS160119till0329' '\mnt\g\kalpaki_ns' 'E:\LATEST_NORTH_HELLENIC_DB'

python3 signalforge.py hel '\mnt\g\Greek Data\Parnon\Raw Data' '\mnt\g\Greek Data\Parnon\Raw Data\Parnon220722  OK' '\mnt\g\MY PASSPORT\DubParnon220722  OK' '\mnt\g\MY PASSPORT\Dubbles PC' '\mnt\g\MY PASSPORT\Dubbles PC\Parnon220722' '\mnt\g\MY PASSPORT\ΑΤΑΞΙΝΟΜΗΤΑ' '\mnt\g\MY PASSPORT\ΑΤΑΞΙΝΟΜΗΤΑ\Parnon220722' '\mnt\g\MY PASSPORT\ΤΑΞΙΝΟΜΗΣΕΙΣ' '\mnt\g\MY PASSPORT\ΤΑΞΙΝΟΜΗΣΕΙΣ' '\mnt\g\Parnon151119' '\mnt\g\Parnon20230706' '\mnt\g\SRD Sata' '\mnt\g\SRD Sata' '\mnt\g\SRD Sata\ΜΕΤΑΦΟΡΑ' '\mnt\g\SRD Sata\ΜΕΤΑΦΟΡΑ\DATASR' '\mnt\g\Semi SOUTH STATION' '\mnt\g\Semi SOUTH STATION\DRAFI' '\mnt\g\Semi SOUTH STATION\PARNON data\Images' '\mnt\g\Semi SOUTH STATION\PARNON data\Measurements' '\mnt\g\Semi SOUTH STATION\ParnonGen' '\mnt\g\Semi SOUTH STATION\SRData' '\mnt\g\TURBO-X\plaisio\Desktop' '\mnt\g\srParnon November 22' '\mnt\g\ΑΡΧΕΙΑ SRD\PARNON Raw Data' 'E:\LATEST_SOUTH_HELLENIC_DB'



#no precursor, light quake
python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20220620/202206200055.pol'

#no precursor, light quake
python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20220427/202204270150.pol' 


python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20211218/202112180515.pol'

python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20211012/202110120920.pol' --no-fit

python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20210304/202103041835.pol' --no-fit

# so much "precursors" that seems like some other noise
python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20210303/202103031015.pol' --no-fit

# so much "precursors" that seems like some other noise
python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20210217/202102170335.pol' --no-fit

python3 py/signal_to_psd.py --file-path '/mnt/e/NEW_POLSKI_DB/20211010/202110101115.pol' --no-fit

python3 py/plot_period_spectograms.py -d '/mnt/e/NEW2_POLSKI_DB' -o '../spectrograms_pol' -t pol

python3 py/signal_to_psd.py --gui -t pol --no-fit


07-27-2024 15:35 LOOKS LIKE PRECURSOR BUT IT AINT BECAUSE QUAKE IS AT CRETE, TOO FAR