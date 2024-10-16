function [p_NS_filtered, f_NS_filtered, p_EW_filtered, f_EW_filtered] = plot_schumann_psd(HNS, HEW, sampling_rate)
    % Compute the PSD for both signals
    [p_NS, f_NS] = pwelch(HNS, [], [], [], sampling_rate);
    [p_EW, f_EW] = pwelch(HEW, [], [], [], sampling_rate);
    
    % Define the frequency range you want to keep (3 Hz to 48 Hz)
    lower_cutoff = 3;
    upper_cutoff = 48;
    
    % Apply the mask to keep frequencies between 3 Hz and 48 Hz
    mask_HNS = (f_NS >= lower_cutoff) & (f_NS <= upper_cutoff);
    mask_HEW = (f_EW >= lower_cutoff) & (f_EW <= upper_cutoff);
    
    % Filter the frequencies and PSD values based on the mask
    f_NS_filtered = f_NS(mask_HNS);
    p_NS_filtered = p_NS(mask_HNS);
    
    f_EW_filtered = f_EW(mask_HEW);
    p_EW_filtered = p_EW(mask_HEW);
    
    % Now you can use the filtered data for further analysis or plotting
    % Plot the filtered PSDs for both HNS and HEW signals
    
    % figure;
    % plot(f_NS_filtered, p_NS_filtered, 'b'); % Plot filtered HNS PSD
    % hold on;
    % plot(f_EW_filtered, p_EW_filtered, 'r'); % Plot filtered HEW PSD
    % hold off;
    % 
    % title('Filtered PSD (3 Hz to 48 Hz) of Downsampled HNS and HEW Signals');
    % xlabel('Frequency (Hz)');
    % ylabel('Power/Frequency (pT^2/Hz)');
    % ylim([0,0.6])
    % legend('Downsampled HNS', 'Downsampled HEW');
end