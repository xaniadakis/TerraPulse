function main()
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
        [HNS, HEW, nr] = read_dat_file(input_dat_file);

        % Calibrate the data
        [calibrated_HNS, calibrated_HEW] = calibrate_HYL(HNS, HEW, nr);
        downsampled_HNS = downsample_signal(calibrated_HNS, downsampling_factor);
        downsampled_HEW = downsample_signal(calibrated_HEW, downsampling_factor);

        % Plot the signals
        % Define parameters
        signal_duration = 300;  % in seconds
        sampling_rate = 100;    % in Hz (100 samples per second)
        downsampled_length = length(downsampled_HNS);  % Assuming both signals have the same length
        
        % Create time vector for the downsampled signals
        time_vector = linspace(0, signal_duration, downsampled_length);
        
        % Plot the signals in linear time space
        figure;
        
        % Plot downsampled HNS signal
        subplot(2, 1, 1);
        plot(time_vector, downsampled_HNS);
        title('Downsampled HNS Signal');
        xlabel('Time (seconds)');
        ylabel('Amplitude');
        
        % Plot downsampled HEW signal
        subplot(2, 1, 2);
        plot(time_vector, downsampled_HEW);
        title('Downsampled HEW Signal');
        xlabel('Time (seconds)');
        ylabel('Amplitude');


    end
end
