function [m, b, mse_history] = gradient_descent_numeric(X, y, alpha, max_iter, tol, method)
    % Gradient Descent with Numerical Gradients
    % method: 'forward' or 'central' difference
    
    n = length(X);
    m = randn();
    b = randn();
    h = 1e-5;  % Small perturbation
    
    mse_history = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        % Current predictions and MSE
        y_pred = m * X + b;
        mse_current = mean((y_pred - y).^2);
        mse_history(iter) = mse_current;
        
        % Numerical gradient for m
        if strcmp(method, 'forward')
            % Forward difference
            mse_m_perturbed = mean(((m + h) * X + b - y).^2);
            dm = (mse_m_perturbed - mse_current) / h;
            
            % Forward difference for b
            mse_b_perturbed = mean((m * X + (b + h) - y).^2);
            db = (mse_b_perturbed - mse_current) / h;
            
        elseif strcmp(method, 'central')
            % Central difference (more accurate)
            mse_m_plus = mean(((m + h) * X + b - y).^2);
            mse_m_minus = mean(((m - h) * X + b - y).^2);
            dm = (mse_m_plus - mse_m_minus) / (2*h);
            
            mse_b_plus = mean((m * X + (b + h) - y).^2);
            mse_b_minus = mean((m * X + (b - h) - y).^2);
            db = (mse_b_plus - mse_b_minus) / (2*h);
        else
            error('Method must be "forward" or "central"');
        end
        
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