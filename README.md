
compile and run signal to zst flow by running the bash script:
 > ./signalforge.sh pol /mnt/f/POLISH\ DATA/Raw\ Data/JanuszNew/20210127 /mnt/e/POLSKI_DB
 > ./signalforge.sh hel /mnt/f/PARNON\ Raw\ Data/JuneAugust\ 2021 /mnt/e/HELLENIC_DB

if you need to plot psd from c (not properly working yet):
 > gnuplot -e "set terminal png; set output '202301060125_psd.png'; set xlabel 'Frequency (Hz)'; set ylabel 'PSD'; set xrange [0:50]; set yrange [0:0.6]; plot './output/202301060125_psd.txt' using 1:2 with lines title 'PSD'"

To convert my code from matlab to C, I run this:
 > codegen lorentzian_fit_psd -args {coder.typeof(0, [10000, 1], [1, 0]), coder.typeof(0, [10000, 1], [1, 0]), 0} -o lorentzian_fit_psd_codegen -config:lib -g

we need to include those headers to potentially run the code:
 > gcc -I/usr/local/MATLAB/R2024a/extern/include -I/home/vag/PycharmProjects/TerraPulse/matlab/codegen/lib/lorentzian_fit_psd main.c -o main

