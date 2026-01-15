#==============================================================================
# 1. REQUIRED LIBRARIES
#==============================================================================
# 'parallel' is a base R package for parallel computation.
library(parallel)

#==============================================================================
# 2. CORE FUNCTION: LOCAL REGRESSION FOR A SINGLE POINT AND SMOOTHER ROW
#==============================================================================
#' @title Compute local polynomial regression and smoother row for a single point.
#' @description This is the core workhorse function that will be parallelized.
#'              For a single evaluation point x_0, it computes the smoothed
#'              value y_hat_0 and the corresponding row of the smoother matrix (s_0)
#'              such that y_hat_0 = s_0 * y_train.
#'
#' @param eval_point A numeric vector for the point (x_0) to estimate the function at.
#' @param X_train A numeric matrix of predictor variables from the training data.
#' @param y_train A numeric vector of the response variable from the training data.
#' @param H A numeric matrix for the bandwidth. Must be symmetric and positive definite.
#' @param degree The degree of the polynomial to fit locally.
#'
#' @return A list containing `prediction` (the estimated value) and `smoother_row`.

local_poly_reg_point <- function(eval_point, X_train, y_train, H, degree) {
  
  # Ensure dimensions match
  if (length(eval_point) != ncol(X_train)) {
    stop("Dimensionality of evaluation point and training data X must match.")
  }
  
  # Calculate differences (u = X_i - x_0)
  diffs <- t(t(X_train) - eval_point)
  
  # Invert and determinant of the bandwidth matrix once
  H_inv <- solve(H)
  detH <- det(H)
  
  # Calculate squared Mahalanobis-like distances for the kernel
  dists_sq <- rowSums((diffs %*% H_inv) * diffs)
  
  # Use the multivariate Gaussian kernel to compute weights
  d <- dim(X_train)[2]
  weights <- (2 * pi)^(-d / 2)*exp(-0.5 * dists_sq)/detH
  
  # Construct the polynomial design matrix (Z)
  Z <- poly(diffs, degree = degree, raw = TRUE)
  Z_intercept <- cbind(1, Z)
  
  # --- Calculate the smoother row s_i ---
  # s_i = e1' * (Z'WZ)^-1 * Z'W, where e1' is a row vector [1, 0, ..., 0]
  # W is conceptually diag(weights), but we use vector operations for efficiency.
  
  # Z'WZ = crossprod(Z_intercept, diag(weights) %*% Z_intercept)
  # A more efficient way is crossprod(Z_intercept, weights * Z_intercept)
  ZTWZ <- crossprod(Z_intercept, weights * Z_intercept)
  
  # Z'W = t(Z_intercept) %*% diag(weights), which is t(weights * Z_intercept)
  ZTW <- t(weights * Z_intercept)
  
  # Calculate (Z'WZ)^-1 * Z'W using the more stable solve(A, B)
  # Use tryCatch for singular matrices which can happen with sparse local data
  smoother_parts <- tryCatch({
    solve(ZTWZ, ZTW)
  }, error = function(e) {
    # If the matrix is singular (not enough local points), return a row of zeros.
    p <- ncol(Z_intercept)
    n <- nrow(X_train)
    matrix(0, nrow = p, ncol = n)
  })
  
  # The smoother row is the first row of this resulting matrix
  smoother_row <- smoother_parts[1, ]
  
  # The prediction is the dot product of the smoother row and y_train
  prediction <- as.numeric(smoother_row %*% y_train)
  
  #return(list(prediction = prediction, smoother_row = smoother_row))
  return(list(prediction = prediction))
}
