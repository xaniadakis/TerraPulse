function params = lorentzian_fit_psd(frequencies, values, numLorentzians)

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
    
    costFunction = @(p) sum((lorentzFunc(p, x) - y).^2);
    params = fminsearch(costFunction, initialGuesses);

    % Display the fitted Lorentzian parameters
    for i = 1:numLorentzians
        g = params((i-1)*3 + 1);      % Amplitude
        x0 = params((i-1)*3 + 2);     % Center frequency
        xi = params((i-1)*3 + 3);     % Width
        fprintf('Lorentzian %d: g = %.4f, x0 = %.4f, xi = %.4f\n', int32(i), g, x0, xi);
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

    % Plot the PSD data and the fitted model
    figure;
    plot(x, y, 'b'); hold on;
    plot(x, lorentzFunc(params, x), 'r--', 'LineWidth', 2); % Increase the LineWidth to make the dotted line thicker
    hold off;
     
    % Add labels
    title(['PSD with Sum of ', num2str(numLorentzians), ' Lorentzian Fits (Balanced)']);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (pT^2/Hz)');
    legend('PSD Data', 'Fitted Model');

    % Return the structure containing fitted parameters
    params = fittedParams; 
end

% Helper function to calculate the sum of Lorentzian functions
function y = sum_lorentzians(p, x, numLorentzians)
    y = zeros(size(x)); % Initialize the output
    for i = 1:numLorentzians
        g = p((i-1)*3 + 1);      % Amplitude
        x0 = p((i-1)*3 + 2);     % Center frequency
        xi = p((i-1)*3 + 3);     % Width
        y = y + g ./ (1 + ((x - x0) / xi).^2); % Sum each Lorentzian
    end
end
