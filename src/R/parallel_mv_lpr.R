#==============================================================================
# 3. PARALLEL WRAPPER FUNCTION
#==============================================================================
#' @title Perform multivariate local polynomial regression in parallel.
#' @description This function distributes the task of calculating local regression
#'              estimates across multiple CPU cores.
#'
#' @param X_eval A matrix of points where the regression function is to be estimated.
#' @param X_train A matrix of predictor variables (training data).
#' @param y_train A vector of the response variable (training data).
#' @param H The bandwidth matrix.
#' @param degree The polynomial degree. Defaults to 1 (local linear).
#' @param n_cores The number of CPU cores to use. If NULL (default), it will
#'                use all available cores minus one.
#'
#' @return A numeric vector of predictions corresponding to each row in `X_eval`.

parallel_mv_lpr <- function(X_eval, X_train, y_train, H, degree = 1, n_cores = NULL) {
  
  # --- Setup Parallel Backend ---
  if (is.null(n_cores)) {
    n_cores <- detectCores() - 1
    if (n_cores < 1) n_cores <- 1
  }
  
  #cat(paste("Setting up parallel cluster with", n_cores, "cores...\n"))
  # 'PSOCK' type works on all operating systems (including Windows)
  #cl <- makeCluster(n_cores, type = "PSOCK")
  
  # --- Export necessary objects and functions to worker nodes ---
  # The workers need the core function and the data to work with.
  #clusterExport(cl, varlist = c("local_poly_reg_point", "X_train", "y_train", "H", "degree"), envir = environment())
  
 
  # --- Distribute the computation ---
  # Convert the evaluation matrix to a list, where each element is a row (a point)
  eval_points_list <- split(X_eval, seq(nrow(X_eval)))
  
  cat("Performing parallel computation...\n")
  # Use parLapply to apply the function to each evaluation point in parallel
  predictions <- parLapply(cl, eval_points_list, function(point) {
    local_poly_reg_point(
      eval_point = point,
      X_train = X_train,
      y_train = y_train,
      H = H,
      degree = degree
    )
  })
  
  
  
  # --- Clean up cluster ---
  #stopCluster(cl)
  #cat("Cluster stopped.\n")
  
  # --- Return results ---
  # unlist converts the list of single predictions back into a vector
  return(unlist(predictions))
}
