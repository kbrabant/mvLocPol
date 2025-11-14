function A = build_design_matrix(X, design_terms)
    % Build design matrix from data and polynomial terms using vectorization.
    %
    % This version is more efficient than a looped approach as it leverages
    % optimized matrix operations.
    %
    % Inputs:
    %   X: n-by-d data matrix (n observations, d variables).
    %   design_terms: p-by-d matrix specifying p polynomial terms.
    %                 Each row defines a term, with entries as powers.
    %
    % Output:
    %   A: n-by-p design matrix.

    [n, d] = size(X);
    p = size(design_terms, 1);
    A = zeros(n, p); % Pre-allocate for efficiency

    % The core of this vectorized approach is to handle all terms and
    % observations simultaneously where possible.
    for j = 1:p
        % For each term (column in A), calculate the product of variables
        % raised to their respective powers.
        % 1. Replicate the j-th row of design_terms 'n' times to match X.
        % 2. Element-wise raise X to the powers defined in the replicated matrix.
        % 3. Take the product across the columns (d variables) to get the
        %    final value for the j-th polynomial term for each observation.
        term_matrix = X .^ repmat(design_terms(j, :), n, 1);
        A(:, j) = prod(term_matrix, 2);
    end
end

