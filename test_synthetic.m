%% test_synthetic.m
% Test gradient descent on synthetic 1D data

fprintf('Generating synthetic data...\n');

% Generate synthetic data
n = 100;
X = linspace(0, 10, n)';  % Feature
m_true = 2.5;
b_true = 5;
noise = randn(n, 1) * 2;  % Gaussian noise
y = m_true * X + b_true + noise;

% Normalize feature
[X_norm, X_mean, X_std] = normalize_feature(X);
y_mean = mean(y);
y_std = std(y);
y_norm = (y - y_mean) / y_std;

fprintf('True relationship: y = %.2f * X + %.2f\n', m_true, b_true);
fprintf('Data shape: %d samples\n', n);

%% Train with Analytic Gradient Descent
fprintf('\nTraining with Analytic Gradient Descent...\n');
alpha = 0.01;
max_iter = 1000;
tol = 1e-6;

tic;
[m_analytic, b_analytic, mse_history_analytic] = gradient_descent_analytic(...
    X_norm, y_norm, alpha, max_iter, tol);
time_analytic = toc;

% Denormalize parameters
m_analytic_orig = m_analytic * (y_std / X_std);
b_analytic_orig = b_analytic * y_std + y_mean - m_analytic_orig * X_mean;

fprintf('  Final m: %.4f (True: %.2f)\n', m_analytic_orig, m_true);
fprintf('  Final b: %.4f (True: %.2f)\n', b_analytic_orig, b_true);
fprintf('  Final MSE: %.6f\n', mse_history_analytic(end));
fprintf('  Iterations: %d\n', length(mse_history_analytic));
fprintf('  Time: %.4f seconds\n', time_analytic);

%% Train with Numerical Gradient Descent (Forward Difference)
fprintf('\nTraining with Numerical Gradient Descent (Forward Difference)...\n');
tic;
[m_numeric_fwd, b_numeric_fwd, mse_history_numeric_fwd] = gradient_descent_numeric(...
    X_norm, y_norm, alpha, max_iter, tol, 'forward');
time_numeric_fwd = toc;

% Denormalize
m_numeric_fwd_orig = m_numeric_fwd * (y_std / X_std);
b_numeric_fwd_orig = b_numeric_fwd * y_std + y_mean - m_numeric_fwd_orig * X_mean;

fprintf('  Final m: %.4f\n', m_numeric_fwd_orig);
fprintf('  Final b: %.4f\n', b_numeric_fwd_orig);
fprintf('  Final MSE: %.6f\n', mse_history_numeric_fwd(end));
fprintf('  Iterations: %d\n', length(mse_history_numeric_fwd));
fprintf('  Time: %.4f seconds\n', time_numeric_fwd);

%% Train with Numerical Gradient Descent (Central Difference)
fprintf('\nTraining with Numerical Gradient Descent (Central Difference)...\n');
tic;
[m_numeric_cent, b_numeric_cent, mse_history_numeric_cent] = gradient_descent_numeric(...
    X_norm, y_norm, alpha, max_iter, tol, 'central');
time_numeric_cent = toc;

% Denormalize
m_numeric_cent_orig = m_numeric_cent * (y_std / X_std);
b_numeric_cent_orig = b_numeric_cent * y_std + y_mean - m_numeric_cent_orig * X_mean;

fprintf('  Final m: %.4f\n', m_numeric_cent_orig);
fprintf('  Final b: %.4f\n', b_numeric_cent_orig);
fprintf('  Final MSE: %.6f\n', mse_history_numeric_cent(end));
fprintf('  Iterations: %d\n', length(mse_history_numeric_cent));
fprintf('  Time: %.4f seconds\n', time_numeric_cent);

%% Plot Results
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Data and regression lines
subplot(1, 3, 1);
plot_regression_line(X, y, m_analytic_orig, b_analytic_orig, ...
    m_numeric_fwd_orig, b_numeric_fwd_orig, m_numeric_cent_orig, b_numeric_cent_orig);
title('Synthetic Data: Fitted Regression Lines');
xlabel('X'); ylabel('y');
legend('Data', 'Analytic GD', 'Numeric (Fwd)', 'Numeric (Cent)', 'Location', 'best');

% Plot 2: Learning curves
subplot(1, 3, 2);
plot_learning_curve(mse_history_analytic, mse_history_numeric_fwd, mse_history_numeric_cent);
title('Learning Curves (MSE vs Iterations)');
xlabel('Iteration'); ylabel('MSE');
legend('Analytic GD', 'Numeric (Fwd)', 'Numeric (Cent)', 'Location', 'best');
grid on;

% Plot 3: Error comparison
subplot(1, 3, 3);
iterations = min([length(mse_history_analytic), length(mse_history_numeric_fwd), ...
                  length(mse_history_numeric_cent)]);
plot(1:iterations, abs(mse_history_analytic(1:iterations) - mse_history_numeric_fwd(1:iterations)), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:iterations, abs(mse_history_analytic(1:iterations) - mse_history_numeric_cent(1:iterations)), 'r-', 'LineWidth', 1.5);
hold off;
title('Gradient Approximation Error');
xlabel('Iteration'); ylabel('|MSE_{analytic} - MSE_{numeric}|');
legend('Forward Difference Error', 'Central Difference Error', 'Location', 'best');
grid on;

% Save plot
saveas(gcf, 'outputs/synthetic_results.png');

%% Display parameter comparison table
fprintf('\nParameter Comparison:\n');
fprintf('-------------------------------------------------\n');
fprintf('Method                 m          b          MSE\n');
fprintf('-------------------------------------------------\n');
fprintf('True                %8.4f   %8.4f       -\n', m_true, b_true);
fprintf('Analytic GD         %8.4f   %8.4f   %8.6f\n', m_analytic_orig, b_analytic_orig, mse_history_analytic(end));
fprintf('Numeric GD (Fwd)    %8.4f   %8.4f   %8.6f\n', m_numeric_fwd_orig, b_numeric_fwd_orig, mse_history_numeric_fwd(end));
fprintf('Numeric GD (Cent)   %8.4f   %8.4f   %8.6f\n', m_numeric_cent_orig, b_numeric_cent_orig, mse_history_numeric_cent(end));
fprintf('-------------------------------------------------\n');