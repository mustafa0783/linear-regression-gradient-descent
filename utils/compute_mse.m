function mse = compute_mse(y_true, y_pred)
    % Compute Mean Squared Error
    mse = mean((y_true - y_pred).^2);
end