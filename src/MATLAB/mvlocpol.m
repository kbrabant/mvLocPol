function [Yhat, L] = mvlocpol(X, y, Xt, options)
% MVLOCPOL:  Fast multivariate local polynomial regression (degrees 0,1,2,3)
% OPTIMIZED VERSION with parallel processing and vectorization
% Usage: [Yhat, L] = mvlocpol(X, y, Xt, options)
%
% Inputs:
%   X   : n x d design points
%   y   : n x 1 responses
%   Xt  : m x d evaluation points
%   options: struct with fields
%       .degree (0..3) default 1
%       .H      dxd bandwidth matrix (required)
%       .k      integer, number of nearest neighbors to use (optional)
%       .radius scalar, radius for neighbor truncation (optional)
%       .kernel {'gauss','epa','bimodgauss','triangle'}; default 'gauss'
%       .regularization  scalar added to Gram diag (default 1e-10)
%       .gpu    logical, default false
%       .parallel logical, use parallel processing (default false)
%       .blockSize evaluation points per block (default 2000)
%       .returnSmoother logical, default false
% Outputs:
% Yhat- Smoothed estimates from local polynomial regression of degree p
% L- Smoother matrix from local polynomial regression of degree p

% Basic checks & defaults
if nargin < 4, options = struct(); end
[n, d] = size(X);
if size(y,1) ~= n, error('X and y dimension mismatch'); end
[m, dt] = size(Xt);
if dt ~= d, error('X and Xt dimension mismatch'); end

degree = getopt(options,'degree',1);
H = getopt(options,'H',[]);
if isempty(H), error('options.H (bandwidth matrix) is required'); end
kernelName = getopt(options,'kernel','gauss');
k = getopt(options,'k',[]);
radius = getopt(options,'radius',[]);
reg = getopt(options,'regularization',1e-10);
useGPU = getopt(options,'gpu',false);
useParallel = getopt(options,'parallel',false);
blockSize = getopt(options,'blockSize',2000);
returnS = getopt(options,'returnSmoother',false);

if isscalar(H)
    H = eye(d)*H;
end

% Precompute constants
invH = H \ eye(d);  % More stable than inv()
detH = max(eps,det(H));
kernelConst = kernel_constants(kernelName, degree, detH);

% Prepare polynomial basis info
[terms, p] = poly_terms(d, degree);
constIdx = find(all(terms==0,2),1);

% Move to GPU if requested (GPU and parallel are mutually exclusive)
if useGPU
    Xg = gpuArray(X);
    yg = gpuArray(y);
    Xtg = gpuArray(Xt);
    invH = gpuArray(invH);
    useParallel = false;  % Can't use both GPU and parallel
else
    Xg = X; yg = y; Xtg = Xt;
end

Yhat = zeros(m,1,'like',Xg);

% Preallocate for smoother matrix
if returnS
    if ~isempty(k) || ~isempty(radius)
        maxNNZ = m * min(n, max(k, 100));
        rows = zeros(maxNNZ, 1);
        cols = zeros(maxNNZ, 1);
        vals = zeros(maxNNZ, 1);
        nnzCount = 0;
    else
        warning('Returning full smoother matrix of size %d x %d (memory heavy).', m, n);
        Lfull = zeros(m,n,'like',Xg);
    end
end

% Precompute neighbor search structure
useNN = ~isempty(k) || ~isempty(radius);
if useNN && ~useGPU
    try
        kd = createns(X,'NSMethod','kdtree');
    catch
        kd = [];
    end
else
    kd = [];
end

% Precompute X * invH for Mahalanobis distances
XinvH = Xg * invH;  % n x d

% Determine number of blocks
numBlocks = ceil(m / blockSize);
blockStarts = 1:blockSize:m;
blockEnds = min(blockStarts + blockSize - 1, m);

% PARALLEL IMPLEMENTATION
if useParallel && ~useGPU
    % Check if parallel pool exists
    p = gcp('nocreate');
    if isempty(p)
        try
            parpool;  % Start default parallel pool
        catch
            warning('Could not start parallel pool. Running serially.');
            useParallel = false;
        end
    end
    
    if useParallel
        % Preallocate cell arrays for parallel blocks
        YhatBlocks = cell(numBlocks, 1);
        if returnS
            rowsBlocks = cell(numBlocks, 1);
            colsBlocks = cell(numBlocks, 1);
            valsBlocks = cell(numBlocks, 1);
        end
        
        % Process blocks in parallel
        parfor blockIdx = 1:numBlocks
            i0 = blockStarts(blockIdx);
            i1 = blockEnds(blockIdx);
            
            % Call helper function for each block
            if returnS
                [YhatBlocks{blockIdx}, rowsBlocks{blockIdx}, ...
                 colsBlocks{blockIdx}, valsBlocks{blockIdx}] = ...
                    process_block(i0, i1, Xg, yg, Xtg, invH, kd, ...
                    kernelName, kernelConst, terms, constIdx, ...
                    reg, k, radius, useNN, true);
            else
                YhatBlocks{blockIdx} = ...
                    process_block(i0, i1, Xg, yg, Xtg, invH, kd, ...
                    kernelName, kernelConst, terms, constIdx, ...
                    reg, k, radius, useNN, false);
            end
        end
        
        % Combine results from parallel blocks
        blockIdx = 1;
        for i0 = blockStarts
            i1 = blockEnds(blockIdx);
            Yhat(i0:i1) = YhatBlocks{blockIdx};
            
            if returnS && exist('nnzCount', 'var')
                newEntries = length(rowsBlocks{blockIdx});
                if newEntries > 0
                    if nnzCount + newEntries > length(rows)
                        rows = [rows; zeros(maxNNZ, 1)];
                        cols = [cols; zeros(maxNNZ, 1)];
                        vals = [vals; zeros(maxNNZ, 1)];
                    end
                    rows(nnzCount+1:nnzCount+newEntries) = rowsBlocks{blockIdx};
                    cols(nnzCount+1:nnzCount+newEntries) = colsBlocks{blockIdx};
                    vals(nnzCount+1:nnzCount+newEntries) = valsBlocks{blockIdx};
                    nnzCount = nnzCount + newEntries;
                end
            elseif returnS && exist('Lfull', 'var')
                % Combine dense smoother rows
                for jj = 1:size(rowsBlocks{blockIdx}, 1)
                    rowIdx = rowsBlocks{blockIdx}(jj);
                    colIdx = colsBlocks{blockIdx}{jj};
                    valIdx = valsBlocks{blockIdx}{jj};
                    Lfull(rowIdx, colIdx) = valIdx;
                end
            end
            blockIdx = blockIdx + 1;
        end
    end
else
    % SERIAL/GPU IMPLEMENTATION (original optimized)
    for i0 = 1:blockSize:m
        i1 = min(m, i0+blockSize-1);
        idxBlock = i0:i1;
        Xte = Xtg(idxBlock, :);
        b = size(Xte,1);
        
        XteinvH = Xte * invH;
        
        if useNN && ~isempty(kd) && ~isempty(k)
            [nnIdx, ~] = knnsearch(kd, gather(Xte), 'K', k);
        else
            nnIdx = [];
        end

        for jj = 1:b
            if useNN
                if ~isempty(nnIdx)
                    idx = nnIdx(jj, :);
                else
                    diffs = Xg - Xte(jj, :);
                    D2 = sum((diffs * invH) .* diffs, 2);
                    
                    if ~isempty(k)
                        [~, idx] = mink(D2, k);
                    elseif ~isempty(radius)
                        idx = find(D2 <= radius^2);
                        if isempty(idx), idx = 1; end
                    else
                        idx = 1:n;
                    end
                end
            else
                idx = 1:n;
            end

            Xi = Xg(idx, :);
            yi = yg(idx);
            
            diffs = Xi - Xte(jj, :);
            arg = sum((diffs * invH) .* diffs, 2);
            w = kernel_weight(arg, kernelName, kernelConst);
            
            if max(w) < 1e-15
                Yhat(i0 + jj - 1) = 0;
                continue;
            end

            Z = eval_poly_basis_fast(diffs, terms, p);
            sqrtW = sqrt(w);
            WZ = Z .* sqrtW;
            Wy = yi .* sqrtW;

            G = WZ' * WZ + reg * eye(p, 'like', WZ);
            rhs = WZ' * Wy;
            beta = G \ rhs;
            
            Yhat(i0 + jj - 1) = beta(constIdx);

            if returnS
                A = G \ (Z' .* w');
                L_row_sel = A(constIdx, :)';
                
                if exist('nnzCount', 'var')
                    newEntries = numel(idx);
                    if nnzCount + newEntries > length(rows)
                        rows = [rows; zeros(maxNNZ, 1)];
                        cols = [cols; zeros(maxNNZ, 1)];
                        vals = [vals; zeros(maxNNZ, 1)];
                    end
                    rows(nnzCount+1:nnzCount+newEntries) = i0 + jj - 1;
                    cols(nnzCount+1:nnzCount+newEntries) = idx;
                    vals(nnzCount+1:nnzCount+newEntries) = gather(L_row_sel);
                    nnzCount = nnzCount + newEntries;
                else
                    Lfull(i0+jj-1, idx) = gather(L_row_sel)';
                end
            end
        end
    end
end
