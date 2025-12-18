function [Hopt,loocv,opt] = cvlocpol(X,y,deg,kernel,Hinit)
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

% Settings for the optimizer
options = optimoptions('fmincon','Display','off','Algorithm','sqp');

% Options settings for the multivariate local polynomial regression
% estimator
opt = struct();
opt.degree = deg;        % 0,1,2,3
opt.H = Hinit;           % dxd bandwidth (required)
opt.kernel = kernel;    % kernel function
opt.k = 200;           % use k nearest neighbors per eval point (optional)
opt.gpu = false;       % true to run on GPU
opt.returnSmoother = true; % true to return L (mxn sparse-ish if k used)
opt.blockSize = 2000;  % evaluation points per block
opt.parallel = 1;

if isscalar(Hinit)
    LB = 0.001; UB = 10;
else
    LB =[0 0; 0 0]; UB =[8 0; 0 8];
end
[Hopt, loocv] = fmincon(@(H)loolocpol(X,y,opt,H),Hinit,[],[],[],[],LB,UB,[],options);
opt.H = Hopt;
opt.returnSmoother = false;

end

function cv = loolocpol(X,y,options,H)
options.H = H;
[Yh, L] = mvlocpol(X, y, X, options);

cv = (1/size(X,1))*sum(((y-Yh)./(1-diag(L))).^2); % LOO-CV
%cv = (1/size(X,1))*((y-Yh)'*(y-Yh)); % RSS
end
