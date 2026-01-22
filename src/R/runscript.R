#==============================================================================
# 4. EXAMPLE USAGE
#==============================================================================

# --- Generate some sample 2D data ---
set.seed(123)
n_train <- 400
n_eval <- 400

# Training data from a noisy sine-cosine surface
X_train <- matrix(runif(n_train * 2, min = -2, max = 2), ncol = 2)
true_surface <- sin(X_train[, 1]) * cos(X_train[, 2])
y_train <- true_surface + rnorm(n_train, mean = 0, sd = 0.3)

# Evaluation data on a regular grid for smooth plotting
grid_points <- seq(-2, 2, length.out = sqrt(n_eval))
X_eval <- expand.grid(x1 = grid_points, x2 = grid_points)


# --- Define Bandwidth Matrix (H) ---
# This is a critical tuning parameter. Here we use a simple diagonal matrix.
# A full matrix H would allow for rotation of the kernel.
h1 <- 0.6
h2 <- 0.6
H <- diag(c(h1^2, h2^2))
#degree <- 3

# -- set up clusters
n_cores <- detectCores()
cat(paste("Setting up parallel cluster with", n_cores, "cores...\n"))
# 'PSOCK' type works on all operating systems (including Windows)
cl <- makeCluster(n_cores, type = "PSOCK")

# --- Export necessary objects and functions to worker nodes ---
# The workers need the core function and the data to work with.
clusterExport(cl, varlist = c("local_poly_reg_point", "X_train", "y_train", "H", "degree"), envir = environment())


# --- Run the parallel regression ---
cat("Starting Multivariate Local Polynomial Regression Example...\n\n")
start_time <- Sys.time()

# Use local linear regression (degree = 1)
predictions <- parallel_mv_lpr(
  X_eval = as.matrix(X_eval),
  X_train = X_train,
  y_train = y_train,
  H = H,
  degree = 3,
  n_cores = 24 # Using 24 cores for this example
)

end_time <- Sys.time()
cat("\nCalculation finished.\n")
print(paste("Time taken:", round(end_time - start_time, 2), "seconds"))


# --- Visualize the results ---
# Reshape predictions into a matrix for plotting
pred_matrix <- matrix(predictions, nrow = sqrt(n_eval))

# Plot the resulting surface
persp(
  x = grid_points,
  y = grid_points,
  z = pred_matrix,
  theta = 30,
  phi = 30,
  expand = 0.6,
  col = "lightblue",
  shade = 0.5,
  ticktype = "detailed",
  xlab = "X1", ylab = "X2", zlab = "Predicted Y",
  main = "Estimated Surface via Parallel Local Regression"
)

cat("\n--- Example Complete ---\n")