function [Hopt,loocv,opt] = cvlocpol(X, y, deg, kernel, Hinit)
%   opt: struct with fields
%       .H      dxd bandwidth matrix (required)
%       .k      integer, number of nearest neighbors to use (optional)
%       .radius scalar, radius for neighbor truncation (optional)
%       .kernel {'gauss','epa','bimodgauss','triangle'} default 'gauss'
%       .regularization  scalar added to Gram diag (default 1e-10)
%       .gpu    logical, default false
%       .parallel logical, use parallel processing (default false)
%       .blockSize evaluation points per block (default 2000)
%       .returnSmoother logical, default false
