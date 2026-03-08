function plot_learning_curve(mse1, mse2, mse3, dataset_name)
    % Plot learning curves
    if nargin < 4
        dataset_name = 'Synthetic Data';
    end
    
    plot(mse1, 'r-', 'LineWidth', 1.5);
    hold on;
    
    if ~isempty(mse2)
        plot(mse2, 'g--', 'LineWidth', 1.5);
    end
    
    if ~isempty(mse3)
        plot(mse3, 'm:', 'LineWidth', 1.5);
    end
    
    hold off;
    xlabel('Iteration');
    ylabel('Mean Squared Error (MSE)');
    title(sprintf('Learning Curve - %s', dataset_name));
    grid on;
    
    if ~isempty(mse3)
        legend('Analytic GD', 'Numeric (Fwd)', 'Numeric (Cent)', 'Location', 'best');
    elseif ~isempty(mse2)
        legend('Analytic GD', 'Numeric GD', 'Location', 'best');
    else
        legend('Analytic GD', 'Location', 'best');
    end
end