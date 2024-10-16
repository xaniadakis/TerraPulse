function main()
    % Prompt the user for input and store the value
    num_lorentzians = input('Enter the num of lorentzians: ');

    % Set parameters
    input_dir = '/media/vag/Users/echan/Documents/Parnon/20230106/';
    sampling_frequency = 5e6 / 128 / 13;
    downsampling_factor = 30;

    % List all .dat files in the input directory
    files = dir(fullfile(input_dir, '*.dat'));
    
    if isempty(files)
        error('No .dat files found in the input directory.');
    end

    i = 1;  % Set the number of iterations you want
    num_files = numel(files);
    
    % Loop through the files but stop after min(i, num_files) iterations
    for k = 1:min(i, num_files)
        % Read the .dat file
        input_dat_file = fullfile(input_dir, files(k).name);
        [HNS, HEW, nr, tini] = ELA11C_ADCread(input_dat_file);

        % Calibrate the data
        [calibrated_HNS, calibrated_HEW] = calibrate_HYL(HNS, HEW, nr);
        downsampled_HNS = downsample_signal(calibrated_HNS, downsampling_factor);
        downsampled_HEW = downsample_signal(calibrated_HEW, downsampling_factor);

        % Print the first 10 samples of the downsampled signals
        fprintf('First 10 samples of downsampled HNS:\n');
        disp(downsampled_HNS(1:min(10, length(downsampled_HNS))));
        
        fprintf('First 10 samples of downsampled HEW:\n');
        disp(downsampled_HEW(1:min(10, length(downsampled_HEW))));

        % Plot the signals
        % Define parameters
        signal_duration = 300;  % in seconds
        sampling_rate = 5e6 / 128 / 13;    % in Hz (100 samples per second)
        
        % plot_schumann_signal(calibrated_HNS, calibrated_HEW, signal_duration)

        [p_NS_filtered, f_NS_filtered, ~, ~] = plot_schumann_psd(calibrated_HNS, calibrated_HEW, sampling_rate);

        lorentzian_fit_psd(f_NS_filtered, p_NS_filtered, num_lorentzians);
        % lorentzian_fit(f_NS_filtered, p_NS_filtered, num_lorentzians);

    end
end
