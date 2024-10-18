function params = lorentzian_fit(frequencies, values, numLorentzians, direction, plotFlag)
    % Frequency and PSD data
    x = frequencies; % Frequency vector
    y = values;      % PSD values (in pT^2/Hz)

    % Define the sum of N Lorentzian functions without using arrayfun
    lorentzFunc = @(p, x) sum_lorentzians(p, x, numLorentzians);

    % Initial guesses for the amplitudes, center frequencies, and widths
    positions = [0.1073, 0.251, 0.3955, 0.54, 0.684, 0.817, 0.95]; % Predefined positions for center frequencies
    initialGuesses = [];
    for i = 1:numLorentzians
        posIndex = mod(i-1, length(positions)) + 1; % Repeat positions if more Lorentzians are used
        initialGuesses = [initialGuesses, max(y)/i, x(ceil(end*positions(posIndex))), 1]; 
    end
    
    % Set lower bounds and upper bounds for the parameters
    % Amplitude (g): >= 0
    % Center frequency (x0): between 3 Hz and 48 Hz
    % Width (xi): >= 0 (no upper bound for width)
    lb = [repmat([0, 3, 0], 1, numLorentzians)];   % Lower bounds: g >= 0, x0 >= 3 Hz, xi >= 0
    ub = [repmat([max(y), 48, Inf], 1, numLorentzians)];  % Upper bounds: g <= max(y), x0 <= 48 Hz, xi is unbounded

    % Exponential Weighting Scheme: More emphasis on peaks and low values
    % weights = y.^1.5; % Exponential scaling for peaks
    % weights(y < mean(y)) = log(1 + y(y < mean(y))); % Apply log scaling for lower values

    % Inverse Weighting - gives more weight to smaller values 
    % weights = 1 ./ y;

    % Square Root Weighting - very slow - reduces the impact of very large values but still maintains some emphasis on higher values
    % weights = sqrt(y);
    
    % Logarithmic Weighting - reduces the influence of larger values 
    % weights = log(1 + y);  % Ensures non-negative values

    % Gaussian Weighting - applies a Gaussian weighting centered at a particular value
    mu = mean(x);  % Center of Gaussian weighting
    sigma = std(x);  % Width of Gaussian weighting
    weights = exp(-((x - mu).^2) / (2 * sigma^2));

    % Residual-Based Weighting - After an initial fit, we can apply weights based on the residuals to improve the fit iteratively
    % residuals = y - fittedModel;  % Calculate residuals from an initial fit
    % weights = 1 ./ (1 + abs(residuals));  % Give less weight to outliers

    % Frequency-Dependent Weighting - emphasizes in certain frequency bands
    % weights = 1 ./ (1 + (x - targetFrequency).^2);  % Emphasize values around a specific fre

    % Perform the curve fitting using lsqcurvefit with weights
    options = optimoptions('lsqcurvefit', 'MaxIterations', 5000, 'FunctionTolerance', 1e-8, 'StepTolerance', 1e-8, 'Display', 'none');
    params = lsqcurvefit(@(p, x) lorentzFunc(p, x).*weights, initialGuesses, x, y.*weights, lb, ub, options);
    
    % Optionally plot the result
    if plotFlag
        figure;
        plot(x, y, 'b'); hold on;
        plot(x, lorentzFunc(params, x), 'r--', 'LineWidth', 2); 
        hold off;
        title([direction, ' PSD with Sum of ', num2str(numLorentzians), ' Lorentzian Fits']);
        xlabel('Frequency (Hz)');
        ylabel('Power/Frequency (pT^2/Hz)');
        legend('PSD Data', 'Fitted Model');
    end

    % Display the fitted Lorentzian parameters
    for i = 1:numLorentzians
        g = params((i-1)*3 + 1);      % Amplitude
        x0 = params((i-1)*3 + 2);     % Center frequency
        xi = params((i-1)*3 + 3);     % Width
        % fprintf('Lorentzian %d: g = %.4f, x0 = %.4f, xi = %.4f\n', i, g, x0, xi);
    end

    % Return the fitted Lorentzian parameters as a structure
    fittedParams = struct('Amplitude', zeros(numLorentzians, 1), ...
                          'CenterFrequency', zeros(numLorentzians, 1), ...
                          'Width', zeros(numLorentzians, 1));

    for i = 1:numLorentzians
        fittedParams.Amplitude(i) = params((i-1)*3 + 1);      % Amplitude
        fittedParams.CenterFrequency(i) = params((i-1)*3 + 2); % Center frequency
        fittedParams.Width(i) = params((i-1)*3 + 3);           % Width
    end

    params = fittedParams;

end


