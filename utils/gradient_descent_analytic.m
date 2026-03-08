function [m, b, mse_history] = gradient_descent_analytic(X, y, alpha, max_iter, tol)
    % Gradient Descent with Analytic Gradients
    % X: normalized features (n x 1)
    % y: normalized target (n x 1)
    % alpha: learning rate
    % max_iter: maximum iterations
    % tol: convergence tolerance
    
    n = length(X);
    m = randn();  % Random initialization
    b = randn();
    
    mse_history = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        % Predictions
        y_pred = m * X + b;
        
        % Compute MSE
        mse = mean((y_pred - y).^2);
        mse_history(iter) = mse;
        
        % Compute gradients analytically
        dm = (2/n) * sum(X .* (y_pred - y));
        db = (2/n) * sum(y_pred - y);
        
        % Update parameters
        m = m - alpha * dm;
        b = b - alpha * db;
        
        % Check convergence
        if iter > 1 && abs(mse_history(iter) - mse_history(iter-1)) < tol
            mse_history = mse_history(1:iter);
            break;
        end
    end
end