function fitResults = plot_combined_lorentzian_fit(freq_NS, psd_NS, freq_EW, psd_EW, numLorentzians, plotFlag)

    % Define the Lorentzian function
    lorentzianSumFunc = @(params, x) sum_lorentzians(params, x, numLorentzians);

    % Predefined positions for center frequencies
    centerFrequencies = [0.1073, 0.251, 0.3955, 0.54, 0.684, 0.817, 0.95]; 

    % Preallocate space for initial parameter guesses
    initialParams_NS = zeros(1, 3 * numLorentzians);
    initialParams_EW = zeros(1, 3 * numLorentzians);

    % Generate initial guesses for both NS and EW data
    for i = 1:numLorentzians
        posIndex = mod(i-1, length(centerFrequencies)) + 1;  % Cycle through positions
        initialParams_NS((i-1)*3 + 1:i*3) = [max(psd_NS)/i, freq_NS(ceil(end*centerFrequencies(posIndex))), 1];
        initialParams_EW((i-1)*3 + 1:i*3) = [max(psd_EW)/i, freq_EW(ceil(end*centerFrequencies(posIndex))), 1];
    end

    % Define lower and upper bounds for parameters
    lowerBounds = [repmat([0, 3, 0], 1, numLorentzians)];   % Lower bounds
    upperBounds_NS = [repmat([max(psd_NS), 48, Inf], 1, numLorentzians)];  % Upper bounds for NS
    upperBounds_EW = [repmat([max(psd_EW), 48, Inf], 1, numLorentzians)];  % Upper bounds for EW

    % Gaussian weighting for NS data (precomputed outside fit)
    mean_NS = mean(freq_NS);
    std_NS = std(freq_NS);
    weights_NS = exp(-((freq_NS - mean_NS).^2) / (2 * std_NS^2));

    % Gaussian weighting for EW data (precomputed outside fit)
    mean_EW = mean(freq_EW);
    std_EW = std(freq_EW);
    weights_EW = exp(-((freq_EW - mean_EW).^2) / (2 * std_EW^2));

    % Set optimization options (loosen tolerances slightly for speed)
    optimOptions = optimoptions('lsqcurvefit', 'MaxIterations', 2000, 'FunctionTolerance', 1e-6, 'StepTolerance', 1e-6, 'Display', 'none');

    % Perform curve fitting for NS and EW data
    params_NS = lsqcurvefit(@(params, x) lorentzianSumFunc(params, x).*weights_NS, initialParams_NS, freq_NS, psd_NS.*weights_NS, lowerBounds, upperBounds_NS, optimOptions);
    params_EW = lsqcurvefit(@(params, x) lorentzianSumFunc(params, x).*weights_EW, initialParams_EW, freq_EW, psd_EW.*weights_EW, lowerBounds, upperBounds_EW, optimOptions);

    if plotFlag
        % Create figure with two subplots
        figure;
    
        % Plot NS direction in the first subplot
        subplot(2, 1, 1);
        plot(freq_NS, psd_NS, 'c'); hold on;
        plot(freq_NS, lorentzianSumFunc(params_NS, freq_NS), 'r--', 'LineWidth', 0.5);
        title(['NS PSD with ', num2str(numLorentzians), ' Lorentzian Fits']);
        xlabel('Frequency (Hz)');
        ylabel('Power/Frequency (pT^2/Hz)');
        legend('NS PSD Data', 'NS Fitted Model');
        hold off;
    
        % Plot EW direction in the second subplot
        subplot(2, 1, 2);
        plot(freq_EW, psd_EW, 'g'); hold on;
        plot(freq_EW, lorentzianSumFunc(params_EW, freq_EW), 'm--', 'LineWidth', 0.5);
        title(['EW PSD with ', num2str(numLorentzians), ' Lorentzian Fits']);
        xlabel('Frequency (Hz)');
        ylabel('Power/Frequency (pT^2/Hz)');
        legend('EW PSD Data', 'EW Fitted Model');
        hold off;
    end

    % Return parameters in a struct
    fitResults = struct('NS_Params', params_NS, 'EW_Params', params_EW);
end
