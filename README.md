
compile and run dat to zst flow by running the bash script:
 > dat_to_zst.sh

if you need to plot psd from c (not properly working yet):
 > gnuplot -e "set terminal png; set output '202301060125_psd.png'; set xlabel 'Frequency (Hz)'; set ylabel 'PSD'; set xrange [0:50]; set yrange [0:200]; plot './output/202301060125_psd.txt' using 1:2 with lines title 'PSD'"


 