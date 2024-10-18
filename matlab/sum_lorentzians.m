function y = sum_lorentzians(p, x, numLorentzians)
    % Reshape p to be a 3 x numLorentzians matrix for easy vectorization
    p = reshape(p, 3, numLorentzians);
    
    % Extract the amplitude (g), center frequency (x0), and width (xi)
    g = p(1, :);    % Amplitudes
    x0 = p(2, :);   % Center frequencies
    xi = p(3, :);   % Widths

    % Calculate all Lorentzians in one go
    % Use broadcasting to create the (x - x0) difference across all x0 values
    y = sum(g ./ (1 + ((x - x0) ./ xi).^2), 2); % Sum across Lorentzians
end
