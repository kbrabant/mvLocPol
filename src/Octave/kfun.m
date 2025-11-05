% Calculates diagonal matrix W = |H|^{-1}K(H^{-1}u) for an arbitrary point xt
% data with Epanechnikov kernel. X is n x p and xt is 1xp
% argument of the kernel K(H^{-1}u)
%
function W = kfun(arg, kernel, const)
switch kernel
    case 'Epa'
        W = const * max(1 - arg, 0);
    case 'gauss'
        W = const * exp(-0.5 * arg);
    case 'bimodgauss'
        W = const * arg .* exp(-arg);
    case 'triangle'
        W = const * max(1 - sqrt(max(arg, 0)), 0); % Fixed sqrt issue
    otherwise
        error('mvlocpol:UnsupportedKernel', 'Unsupported kernel: %s', kernel);
end
end

