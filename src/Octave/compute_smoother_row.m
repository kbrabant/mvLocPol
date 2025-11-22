function L = compute_smoother_row(eval_idx, X, X_eval, H, kernel_type, design_terms, regularization, detH, invH, kernelConst, Ireg, nz)
    % Compute one row of the smoother matrix (weights for one evaluation point)

    xi = X_eval(eval_idx, :);
    diffs = X - xi;

    % Mahalanobis distance: arg = sum((diffs / H) .* diffs, 2)
    scaled_diffs = diffs * invH;
    arg = sum(scaled_diffs .* diffs, 2);
    W = kfun(arg, kernel_type, kernelConst);   % kernel calculation

    % Build design matrix Xx for local polynomial regression up to degree 3
    Xx = build_design_matrix(diffs, design_terms);

    % Calculate the local polynomial estimate for the current point
    % Weighted matrices (no diag)
    %WXx = bsxfun(@times, Xx, W);
    WXx = Xx .* W;

    A = WXx' * Xx + Ireg;
    B = WXx';

    % Cholesky solve (much faster than backslash if SPD)
    % Fallback to A\B if not SPD
    try
        R = chol(A);
        beta = R \ (R' \ B);
    catch
        beta = A \ B;
    end
    L = beta(1,:);
end

