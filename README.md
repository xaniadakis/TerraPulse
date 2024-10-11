
compile and run c program with following command:
 > gcc -o dat_to_text dat_to_text.c -lfftw3 -lm
 >
 > ./dat_to_text

if you need to plot psd from c (not properly working yet):
 > gnuplot -e "set terminal png; set output 'psd_plot.png'; set xlabel 'Frequency (Hz)'; set ylabel 'PSD'; set yrange [0:50]; plot '202301060000_psd.txt' using 1:2 with lines title 'PSD'"


 