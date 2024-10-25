function plot_file(dt, t, x, y, F, Pxx, Pyy, L1, L2)

st = datestr(dt, 'dd mmm yyyy HH:MM:SS');

figure(1);
subplot(2,1,1);
plot(t,x);
grid on;
title([st, ' (N-S)']);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
ylim([-2,2]);
subplot(2,1,2);
plot(F,Pxx,F,L1);
grid on;
xlabel('Frequency (Hz)');
ylabel('PSD (pT^2/Hz)');
% ylabel('PSD (a.u.)');
legend('Raw Spectrum','Lorentzian Fit');

if ~isempty(Pyy) && ~isempty(L2)
figure(2);
subplot(2,1,1);
plot(t,y);
grid on;
title([st, ' (E-W)']);
xlabel('Time (sec)');
ylabel('Amplitude (V)');
ylim([-2,2]);
subplot(2,1,2);
plot(F,Pyy,F,L2);
grid on;
xlabel('Frequency (Hz)');
ylabel('PSD (pT^2/Hz)');
% ylabel('PSD (a.u.)');
legend('Raw Spectrum','Lorentzian Fit');
end


end

