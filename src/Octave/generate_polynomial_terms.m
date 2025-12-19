
function poly_terms = generate_polynomial_terms(d, p)
% GENERATE_POLYNOMIAL_TERMS Generates all polynomial terms up to degree p
%
% Inputs:
%   d - number of dimensions
%   p - maximum polynomial degree
%
% Output:
%   poly_terms - matrix where each row represents powers for each variable

if d == 1
    poly_terms = (0:p)';
    return;
end

% Generate all combinations of powers that sum to at most p
poly_terms = [];
for degree = 0:p
    % Generate all non-negative integer solutions to x1 + x2 + ... + xd = degree
    terms = generate_partitions(d, degree);
    poly_terms = [poly_terms; terms];
end

function partitions = generate_partitions(d, n)
% GENERATE_PARTITIONS Generates all ways to distribute n among d variables
%
% Inputs:
%   d - number of variables
%   n - total degree
%
% Output:
%   partitions - matrix of all valid partitions

if d == 1
    partitions = n;
    return;
end

partitions = [];
for i = 0:n
    sub_partitions = generate_partitions(d-1, n-i);
    partitions = [partitions; [i * ones(size(sub_partitions, 1), 1), sub_partitions]];
end
