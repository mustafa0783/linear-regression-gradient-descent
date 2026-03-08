function [X_norm, X_mean, X_std] = normalize_feature(X)
    % Z-score normalization
    X_mean = mean(X);
    X_std = std(X);
    
    if X_std == 0
        X_std = 1;  % Avoid division by zero
    end
    
    X_norm = (X - X_mean) / X_std;
end