function plot_regression_line(X, y, m1, b1, m2, b2, m3, b3)
    % Plot data and multiple regression lines
    scatter(X, y, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    
    x_line = linspace(min(X), max(X), 100);
    
    if nargin >= 4
        y_line1 = m1 * x_line + b1;
        plot(x_line, y_line1, 'r-', 'LineWidth', 2);
    end
    
    if nargin >= 6
        y_line2 = m2 * x_line + b2;
        plot(x_line, y_line2, 'g--', 'LineWidth', 2);
    end
    
    if nargin >= 8
        y_line3 = m3 * x_line + b3;
        plot(x_line, y_line3, 'm:', 'LineWidth', 2);
    end
    
    hold off;
    grid on;
end