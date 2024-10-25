function main()
    % Main function for processing SRD data
    %
    % This function coordinates the use of several helper functions
    % to read, process, and analyze SRD data from a file.

    % Step 1: Select the file using open_file_dialog
    fpath = open_file_dialog();
    if isempty(fpath)
        disp('No file selected, exiting...');
        return;
    end

    % Step 2: Extract metadata and raw data from the file
    [t, fs, x, y] = get_srd_data(fpath);

    if isempty(x)
        disp('No data available in the file.');
        return;
    end

    % Step 3: Apply equalizer settings based on frequency and date
    % Assuming freqs are derived or calculated elsewhere, we pass it along with the date
    freqs = linspace(0, fs/2, length(x)); % Example of generating frequency range
    [sreq1, sreq2] = get_equalizer(freqs, t);

    % Step 4: Compute spectral density using srd_spec
    [F, Pxx, Pyy, L1, L2, R1, R2, gof1, gof2] = srd_spec(t, fs, x, y);

    % Step 5: Apply resonance fitting using sr_fit
    modes = 6; % Number of modes to fit
    [fitline, noiseline, results, gof] = sr_fit(F, Pxx, modes);

    % Step 6: Plot the results
    plot_file(fpath, F, Pxx, fitline, noiseline);

    % Step 7: Print results of the fitting process
    disp('Resonance Fitting Results:');
    disp(results);
end
