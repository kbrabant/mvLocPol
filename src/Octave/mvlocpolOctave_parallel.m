function [Yhat,L] = mvlocpolOctave_parallel(X,y, deg, kernel, H, varargin)
% Performs Multivariate Local Kernel regression with degree 1, 2 or 3
%  Input:
%     - X = n x d data matrix
%     - y = n x 1 ouput data vector
%     -
%     - deg = degree of the local polynomial estimator
%     - kernel = choice of kernel function between Epa, Gauss, bimodgauss
%     - H = bandwidth or bandwidth matrix for the kernel. If the bandwidh
%       is a scalar, the bandwidth will be set to H*eye(d)
%
% Output:
%     - Yhat = local polynomial estimator for the regression function
%     - L = smoother matrix
%
% Use   Yhat = mvlocpol(X,y, deg, kernel, H)
%       Yhat = mvlocpol(X,y, deg, kernel, H, Xt)
%       with Xt is the data nxd to be evaluated. If not supplied Xt = X


[n,d] = size(X); %Get dimensions of the input data
% Check whether input data and output match
if n ~= length(y)
    error('Input data and output vector must have the same number of observations.');
end
if size(H,1) ~= size(X,2)
    % H will set to H*eye(d)
    H = H * eye(d);
end

if isempty(varargin)
    Xt = X; % Default value for Xt if not supplied
else
    Xt = varargin{1}; % Assign Xt from varargin if supplied
end

[nt,dt] = size(Xt);
if d~=dt
    error('Input dimensions do not match.');
end

% ------------ Precomputations ------------
design_terms = generate_polynomial_terms(d, deg);% Polynomial powers
nz = size(design_terms, 1);             % Number of polynomial terms

% Kernel constants
detH = det(H);
invH = H \ eye(d);                      % Avoid inv(H)
kernelConst = kernel_constants(kernel, d, detH);

% ------------ Main Loop (Parallelized) ------------
Ireg = eye(nz) * 1e-8;  % Regularization


% Parallel computation of smoother matrix rows using pararrayfun
Smoother_rows = pararrayfun(24, @(i)compute_smoother_row(i, X, Xt, H, kernel,design_terms, 1e-8, detH, invH, kernelConst, Ireg, nz), 1:nt, 'UniformOutput', false);

% Convert cell array to matrix
L = vertcat(Smoother_rows'{:});

% Calculate the final local polynomial estimate
Yhat = L * y;



