%% train_real.m
% Train linear regression on real data (Square_Footage vs House_Price)

fprintf('Loading real dataset...\n');

% Load CSV data
data = readtable('data/house_price_data.csv');

% Extract features: Square_Footage vs House_Price
X = data.Square_Footage;
y = data.House_Price;

fprintf('Dataset loaded: %d samples\n', length(X));
fprintf('Features: Square_Footage vs House_Price\n');
fprintf('Price range: $%.2f to $%.2f\n', min(y), max(y));

%% Split into train (159) and test (40)
train_ratio = 159/199;
n_total = length(X);
n_train = 159;
n_test = 40;

% Random split with fixed seed for reproducibility
rng(42);
idx = randperm(n_total);
train_idx = idx(1:n_train);
test_idx = idx(n_train+1:end);

X_train = X(train_idx);
y_train = y(train_idx);
X_test = X(test_idx);
y_test = y(test_idx);

fprintf('\nData split:\n');
fprintf('  Training samples: %d\n', n_train);
fprintf('  Test samples: %d\n', n_test);

%% Normalize training data
[X_train_norm, X_mean, X_std] = normalize_feature(X_train);
[y_train_norm, y_mean, y_std] = normalize_feature(y_train);

% Normalize test data using training statistics
X_test_norm = (X_test - X_mean) / X_std;
y_test_norm = (y_test - y_mean) / y_std;

%% Train with Analytic Gradient Descent
fprintf('\nTraining with Analytic Gradient Descent...\n');
alpha = 0.01;
max_iter = 1000;
tol = 1e-6;

tic;
[m_analytic, b_analytic, mse_history_analytic] = gradient_descent_analytic(...
    X_train_norm, y_train_norm, alpha, max_iter, tol);
time_analytic = toc;

% Denormalize parameters
m_analytic_orig = m_analytic * (y_std / X_std);
b_analytic_orig = b_analytic * y_std + y_mean - m_analytic_orig * X_mean;

fprintf('  Final slope (m): %.4f\n', m_analytic_orig);
fprintf('  Final intercept (b): %.4f\n', b_analytic_orig);
fprintf('  Final MSE: %.6f\n', mse_history_analytic(end));
fprintf('  Iterations: %d\n', length(mse_history_analytic));
fprintf('  Time: %.4f seconds\n', time_analytic);

%% Train with Numerical Gradient Descent (Central Difference - more accurate)
fprintf('\nTraining with Numerical Gradient Descent (Central Difference)...\n');
tic;
[m_numeric, b_numeric, mse_history_numeric] = gradient_descent_numeric(...
    X_train_norm, y_train_norm, alpha, max_iter, tol, 'central');
time_numeric = toc;

% Denormalize
m_numeric_orig = m_numeric * (y_std / X_std);
b_numeric_orig = b_numeric * y_std + y_mean - m_numeric_orig * X_mean;

fprintf('  Final slope (m): %.4f\n', m_numeric_orig);
fprintf('  Final intercept (b): %.4f\n', b_numeric_orig);
fprintf('  Final MSE: %.6f\n', mse_history_numeric(end));
fprintf('  Iterations: %d\n', length(mse_history_numeric));
fprintf('  Time: %.4f seconds\n', time_numeric);

%% Make predictions on test set
% Analytic GD predictions
y_pred_analytic_norm = m_analytic * X_test_norm + b_analytic;
y_pred_analytic = y_pred_analytic_norm * y_std + y_mean;

% Numeric GD predictions
y_pred_numeric_norm = m_numeric * X_test_norm + b_numeric;
y_pred_numeric = y_pred_numeric_norm * y_std + y_mean;

%% Calculate performance metrics
% RMSE
rmse_analytic = sqrt(mean((y_test - y_pred_analytic).^2));
rmse_numeric = sqrt(mean((y_test - y_pred_numeric).^2));

% R-squared
ss_total = sum((y_test - mean(y_test)).^2);
ss_res_analytic = sum((y_test - y_pred_analytic).^2);
ss_res_numeric = sum((y_test - y_pred_numeric).^2);
r2_analytic = 1 - (ss_res_analytic / ss_total);
r2_numeric = 1 - (ss_res_numeric / ss_total);

%% Display performance comparison
fprintf('\nPerformance on Test Set:\n');
fprintf('------------------------------------------------------------\n');
fprintf('Metric                Analytic GD      Numeric GD (Cent)\n');
fprintf('------------------------------------------------------------\n');
fprintf('RMSE                $%12.2f   $%12.2f\n', rmse_analytic, rmse_numeric);
fprintf('R-squared               %8.4f        %8.4f\n', r2_analytic, r2_numeric);
fprintf('Training Time (s)       %8.4f        %8.4f\n', time_analytic, time_numeric);
fprintf('------------------------------------------------------------\n');

%% Plot Results
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Training data with regression lines
subplot(2, 3, 1);
scatter(X_train, y_train, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;

% Generate line points
x_line = linspace(min(X_train), max(X_train), 100);
y_line_analytic = m_analytic_orig * x_line + b_analytic_orig;
y_line_numeric = m_numeric_orig * x_line + b_numeric_orig;

plot(x_line, y_line_analytic, 'r-', 'LineWidth', 2);
plot(x_line, y_line_numeric, 'g--', 'LineWidth', 2);
hold off;

xlabel('Square Footage'); ylabel('House Price ($)');
title('Training Data with Fitted Lines');
legend('Training Data', 'Analytic GD', 'Numeric GD', 'Location', 'best');
grid on;

% Plot 2: Test predictions vs actual
subplot(2, 3, 2);
scatter(y_test, y_pred_analytic, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(y_test, y_pred_numeric, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.6);
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 1.5);
hold off;

xlabel('Actual Price ($)'); ylabel('Predicted Price ($)');
title('Test Set: Predicted vs Actual');
legend('Analytic GD', 'Numeric GD', 'Perfect Prediction', 'Location', 'best');
axis equal;
grid on;

% Plot 3: Residuals
subplot(2, 3, 3);
residuals_analytic = y_test - y_pred_analytic;
residuals_numeric = y_test - y_pred_numeric;

scatter(y_pred_analytic, residuals_analytic, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(y_pred_numeric, residuals_numeric, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.6);
plot([min(y_pred_analytic), max(y_pred_analytic)], [0, 0], 'k--', 'LineWidth', 1.5);
hold off;

xlabel('Predicted Price ($)'); ylabel('Residual ($)');
title('Residual Plot');
legend('Analytic GD', 'Numeric GD', 'Zero Error', 'Location', 'best');
grid on;

% Plot 4: Learning curves
subplot(2, 3, 4);
plot_learning_curve(mse_history_analytic, mse_history_numeric, [], 'Real Data');
title('Learning Curves');
xlabel('Iteration'); ylabel('MSE');
grid on;

% Plot 5: Parameter convergence
subplot(2, 3, 5);
m_history_analytic = m_analytic_orig * ones(size(mse_history_analytic));
m_history_numeric = m_numeric_orig * ones(size(mse_history_numeric));

plot(1:length(mse_history_analytic), m_history_analytic, 'r-', 'LineWidth', 2);
hold on;
plot(1:length(mse_history_numeric), m_history_numeric, 'g--', 'LineWidth', 2);
hold off;

xlabel('Iteration'); ylabel('Slope (m)');
title('Parameter Convergence');
legend('Analytic GD', 'Numeric GD', 'Location', 'best');
grid on;

% Plot 6: Error distribution
subplot(2, 3, 6);
histogram(residuals_analytic, 20, 'FaceColor', 'r', 'FaceAlpha', 0.6);
hold on;
histogram(residuals_numeric, 20, 'FaceColor', 'g', 'FaceAlpha', 0.6);
hold off;

xlabel('Residual ($)'); ylabel('Frequency');
title('Error Distribution');
legend('Analytic GD', 'Numeric GD', 'Location', 'best');
grid on;

% Save plot
saveas(gcf, 'outputs/real_data_results.png');

%% Display regression equation
fprintf('\nRegression Equations:\n');
fprintf('Analytic GD: Price = %.2f * Square_Footage + %.2f\n', m_analytic_orig, b_analytic_orig);
fprintf('Numeric GD:  Price = %.2f * Square_Footage + %.2f\n', m_numeric_orig, b_numeric_orig);

%% Interpretation
fprintf('\nInterpretation:\n');
fprintf('• Each additional square foot increases house price by approximately $%.2f\n', m_analytic_orig);
fprintf('• The base price (for 0 sq ft) is $%.2f\n', b_analytic_orig);
fprintf('• R² = %.4f means %.2f%% of price variation is explained by square footage\n', ...
    r2_analytic, r2_analytic*100);