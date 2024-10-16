function plot_schumann_signal(HNS, HEW, duration)
    % Create time vector for the downsampled signals
    timespace = linspace(0, duration, length(HNS));
    
    % Plot the signals in linear time space
    figure;
    
    % Plot both downsampled HNS and HEW signals in the same plot with different colors
    plot(timespace, HNS, 'b'); % Plot HNS in blue
    hold on;
    plot(timespace, HEW, 'r'); % Plot HEW in red
    hold off;
    
    title('Downsampled HNS and HEW Signals');
    xlabel('Time (seconds)');
    ylabel('Amplitude');
    ylim([-200, 0]); % Set y-axis limits
    
    legend('Downsampled HNS', 'Downsampled HEW'); % Add a legend to distinguish between the two signals

end