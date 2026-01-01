function [Yhat, L] = mvlocpolOctave(X, y, Xt, options)
% Maximum performance multivariate local polynomial regression for Octave
% Extreme optimizations: vectorized batch solving, minimal allocations

if nargin < 4, options = struct(); end
[n, d] = size(X);
[m, dt] = size(Xt);
if dt ~= d, error('X and Xt dimension mismatch'); end
if size(y,1) ~= n, error('X and y dimension mismatch'); end

degree = getopt(options,'degree',1);
H = getopt(options,'H',[]);
if isempty(H), error('options.H (bandwidth matrix) is required'); end
kernelName = getopt(options,'kernel','gauss');
k = getopt(options,'k',[]);
reg = getopt(options,'regularization',1e-10);
blockSize = getopt(options,'blockSize',min(500, m));  % Smaller blocks for cache
returnS = getopt(options,'returnSmoother',false);

if isscalar(H), H = eye(d)*H; end
invH = H \ eye(d);
detH = max(eps, det(H));

[terms, p] = poly_terms_fast(d, degree);
constIdx = find(all(terms==0,2),1);

Yhat = zeros(m,1);
if returnS
    rows = zeros(m*k,1);
    cols = zeros(m*k,1);
    vals = zeros(m*k,1);
    nnzCount = 0;
else
    L = [];
end

kNN = min(k,n);
invHdiag = diag(invH);

% Kernel setup
switch lower(kernelName)
    case 'epa', cd = ((d+2)*gamma(d/2+1))/(2*pi^(d/2)); ktype = 1;
    case 'gauss', cd = (2*pi)^(-d/2); ktype = 0;
    case 'bimodgauss', cd = 2/(pi^(d/2)*d); ktype = 2;
    case 'triangle', cd = d*(d+1)*gamma(d/2)/(2*pi^(d/2)); ktype = 3;
    otherwise, cd = (2*pi)^(-d/2); ktype = 0;
end
kernelConst = cd / detH;

% Pre-compute polynomial evaluation structure
[poly_dims, poly_exps, poly_nd] = build_poly_lookup(terms, p);

% Memory pools
numChunks = ceil(m / blockSize);
regVec = reg * ones(p,1);  % For fast diagonal addition

for chunkIdx = 1:numChunks
    i0 = (chunkIdx-1)*blockSize + 1;
    i1 = min(i0 + blockSize - 1, m);
    b = i1 - i0 + 1;
    Xte = Xt(i0:i1, :);
    totalN = kNN * b;

    % --- Optimized distance computation ---
    D2 = zeros(n, b);
    for dim = 1:d
        D2 += (X(:,dim) - Xte(:,dim)').^2 * invHdiag(dim);
    end

    % --- Fast sorting (partial sort would be better but not in base Octave) ---
    [~, sortIdx] = sort(D2, 1);
    neighborsIdx = sortIdx(1:kNN, :);

    % --- Vectorized neighbor gathering ---
    neighborsFlat = neighborsIdx(:);  % kNN*b x 1
    Xneigh = X(neighborsFlat, :);     % kNN*b x d
    yneigh = y(neighborsFlat);        % kNN*b x 1

    % Compute all differences at once
    XteRep = repelem(Xte, kNN, 1);
    diffsAll = Xneigh - XteRep;

    % --- Ultra-fast polynomial evaluation ---
    Zall = ones(totalN, p);
    for jj = 2:p
        nd = poly_nd(jj);
        dims = poly_dims{jj};
        exps = poly_exps{jj};

        if nd == 1
            Zall(:,jj) = diffsAll(:,dims) .^ exps;
        elseif nd == 2
            Zall(:,jj) = (diffsAll(:,dims(1)) .^ exps(1)) .* ...
                         (diffsAll(:,dims(2)) .^ exps(2));
        else
            temp = ones(totalN, 1);
            for idx = 1:nd
                temp .*= diffsAll(:,dims(idx)) .^ exps(idx);
            end
            Zall(:,jj) = temp;
        end
    end

    % --- Vectorized kernel weights ---
    argAll = sum((diffsAll * invH) .* diffsAll, 2);

    switch ktype
        case 0  % Gauss
            wAll = kernelConst * exp(-0.5 * argAll);
        case 1  % Epanechnikov
            wAll = kernelConst * max(0, 1-argAll);
        case 2  % Bimodal Gauss
            wAll = kernelConst * argAll .* exp(-argAll);
        case 3  % Triangle
            wAll = kernelConst * max(1-sqrt(argAll), 0);
    end

    sqrtWall = sqrt(wAll);

    % --- Block-vectorized regression ---
    % Pre-weight everything
    WZall = Zall .* sqrtWall;
    Wyall = yneigh .* sqrtWall;

    % Reshape for batch processing
    WZblock = reshape(WZall, kNN, b, p);
    Wyblock = reshape(Wyall, kNN, b);
    wblock = reshape(wAll, kNN, b);

    % Vectorized Gram matrices for all b points
    % G(:,:,jj) = WZblock(:,jj,:)' * WZblock(:,jj,:)
    for jj = 1:b
        WZ = reshape(WZblock(:,jj,:), kNN, p);
        Wy = Wyblock(:,jj);

        % Fast Gram matrix
        G = WZ' * WZ;
        G(1:p+1:end) += reg;

        % Solve
        beta = G \ (WZ' * Wy);
        Yhat(i0+jj-1) = beta(constIdx);

        if returnS
            w_local = wblock(:,jj);
            A = G \ (WZ' .* w_local');
            L_row = A(constIdx,:)';
            idx_start = nnzCount + 1;
            idx_end = nnzCount + kNN;
            rows(idx_start:idx_end) = i0+jj-1;
            cols(idx_start:idx_end) = neighborsIdx(:,jj);
            vals(idx_start:idx_end) = L_row;
            nnzCount = idx_end;
        end
    end
end

if returnS
    L = sparse(rows(1:nnzCount), cols(1:nnzCount), vals(1:nnzCount), m, n);
end
end

%% --- Ultra-optimized polynomial helpers ---
function [poly_dims, poly_exps, poly_nd] = build_poly_lookup(terms, p)
    poly_dims = cell(p,1);
    poly_exps = cell(p,1);
    poly_nd = zeros(p,1);

    for jj = 1:p
        nz_idx = find(terms(jj,:) > 0);
        poly_dims{jj} = nz_idx;
        poly_exps{jj} = terms(jj, nz_idx);
        poly_nd(jj) = length(nz_idx);
    end
end

function [terms, p] = poly_terms_fast(d, deg)
    % Optimized polynomial term generation
    if deg == 0
        terms = zeros(1,d);
        p = 1;
        return;
    end

    % Fast paths for common cases
    if d == 1
        terms = (0:deg)';
        p = deg + 1;
        return;
    end

    if d == 2
        % Direct generation for 2D
        terms = zeros((deg+1)*(deg+2)/2, 2);
        idx = 1;
        for total = 0:deg
            for i = 0:total
                terms(idx,:) = [i, total-i];
                idx++;
            end
        end
        p = idx - 1;
        return;
    end

    % General case - iterative construction
    p_estimate = nchoosek(d+deg, deg);
    terms = zeros(p_estimate, d);
    idx = 1;

    % Use queue-based approach
    queue = zeros(p_estimate, d+1);  % [term, remaining_degree]
    queue(1,:) = [zeros(1,d), deg];
    qstart = 1;
    qend = 1;

    terms(1,:) = zeros(1,d);
    idx = 2;

    while qstart <= qend
        current = queue(qstart, 1:d);
        rem_deg = queue(qstart, d+1);
        qstart++;

        if rem_deg > 0
            for dim = 1:d
                new_term = current;
                new_term(dim)++;

                % Check if already seen or can be added
                can_add = true;
                for prev = 1:idx-1
                    if all(terms(prev,:) == new_term)
                        can_add = false;
                        break;
                    end
                end

                if can_add && sum(new_term) <= deg
                    terms(idx,:) = new_term;
                    idx++;
                    if rem_deg > 1
                        qend++;
                        queue(qend,:) = [new_term, rem_deg-1];
                    end
                end
            end
        end
    end

    % Fallback to recursive if queue fails
    if idx == 2
        terms = [];
        for t = 0:deg
            terms = [terms; compose_vec_fast(d, t)];
        end
        p = size(terms, 1);
    else
        terms = terms(1:idx-1,:);
        p = idx - 1;
    end
end

function C = compose_vec_fast(d, total)
    if d == 1
        C = total;
        return;
    end
    if total == 0
        C = zeros(1,d);
        return;
    end

    % Iterative composition
    C = [];
    for k = 0:total
        tail = compose_vec_fast(d-1, total-k);
        C = [C; repmat(k, size(tail,1), 1), tail];
    end
end

function val = getopt(s, name, default)
    if isfield(s,name), val = s.(name); else, val = default; end
end
