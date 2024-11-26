function main(numLorentzians, fileSuffix, inputDir)
    % Check if required arguments are provided
    if nargin < 2
        error('You must provide the number of Lorentzians and the file type as command-line arguments.');
    end

    % Set parameters
    % input_dir = '../output/';
    sampling_frequency = 5e6 / 128 / 13;
    downsampling_factor = 30;
    downsampling_rate = sampling_frequency / downsampling_factor;

    % Validate the input directory
    if ~isfolder(inputDir)
        error('The specified input directory does not exist: %s', inputDir);
    end

    % List all .txt files in the input directory
    files = dir(fullfile(inputDir, ['*.', fileSuffix]));
    if isempty(files)
        error(['No .', fileSuffix, ' files found in the input directory: ', inputDir]);
    end

    i = Inf;  % Set the number of iterations you want (or the number of files to process)
    num_files = numel(files);
    
    tic;
    % Process up to i files
    for k = 1:min(i, num_files)
        % Display progress percentage
        fprintf(1, '\rProcessing file %d of %d (%.2f%%)', k, min(i, num_files), (k / min(i, num_files)) * 100);
        
        % Load data from the .txt file
        input_filename = fullfile(inputDir, files(k).name);
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
        save(fullfile(inputDir, [file_name, '.mat']), 'fitResults', '-v7.3');  % Save as compressed .mat file
    end
    toc;
    num_files
    
    fprintf('Processing complete.\n');
end
