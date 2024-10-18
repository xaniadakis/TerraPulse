function main(numLorentzians)
    % Check if num_lorentzians is provided
    if nargin < 1
        error('You must provide the number of Lorentzians as a command-line argument.');
    end

    % Set parameters
    input_dir = '../output/';
    sampling_frequency = 5e6 / 128 / 13;
    downsampling_factor = 30;
    downsampling_rate = sampling_frequency / downsampling_factor;

    % List all .txt files in the input directory
    files = dir(fullfile(input_dir, '*.txt'));
    if isempty(files)
        error('No .txt files found in the input directory.');
    end

    i = Inf;  % Set the number of iterations you want (or the number of files to process)
    num_files = numel(files);
    
    tic;
    % Process up to i files
    for k = 1:min(i, num_files)
        % Display progress percentage
        fprintf(1, '\rProcessing file %d of %d (%.2f%%)', k, min(i, num_files), (k / min(i, num_files)) * 100);
        
        % Load data from the .txt file
        input_filename = fullfile(input_dir, files(k).name);
        data = load(input_filename);  % Load the data from the file
        % data = dlmread(input_filename, '\n', 2, 0);  % Read data, skipping the first two lines

        downsampled_HNS = data(:, 1);  % First column
        downsampled_HEW = data(:, 2);  % Second column

        % Compute the power spectral density (PSD)
        [p_NS_filtered, f_NS_filtered, p_EW_filtered, f_EW_filtered] = plot_schumann_psd(downsampled_HNS, downsampled_HEW, downsampling_rate);

        % Perform the Lorentzian fit and plot the results
        fitResults = plot_combined_lorentzian_fit(f_NS_filtered, p_NS_filtered, f_EW_filtered, p_EW_filtered, numLorentzians, false);

        % Save fitResults in a compressed .mat file with the same name as the input .txt file
        [~, file_name, ~] = fileparts(files(k).name);  % Extract the file name without extension
        save(fullfile(input_dir, [file_name, '.mat']), 'fitResults', '-v7.3');  % Save as compressed .mat file
    end
    toc;
    num_files
    
    fprintf('Processing complete.\n');
end
