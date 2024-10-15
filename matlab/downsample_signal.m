function downsampled_signal = downsample_signal(input_signal, downsampling_factor)
    % Calculate the downsampled length
    downsampled_length = floor(length(input_signal) / downsampling_factor);
    
    % Preallocate the downsampled signal
    downsampled_signal = zeros(1, downsampled_length);

    % Perform downsampling by averaging blocks of DOWNSAMPLING_FACTOR samples
    for i = 1:downsampled_length
        % Calculate the average of the current block
        avg_value = mean(input_signal((i-1) * downsampling_factor + 1 : i * downsampling_factor));
        % Store the average in the downsampled signal
        downsampled_signal(i) = avg_value;
    end
end
