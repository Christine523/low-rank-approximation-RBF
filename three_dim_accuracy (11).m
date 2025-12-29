function main()
    % --- 1. Setup ---
    rng(0); % Set random seed for reproducibility
    
    % Training size configuration
    training_sizes_pure = [64, 128, 256, 512, 1024, 2048, 4096, 8192];  % Pure methods
    training_sizes_nystrom = [512, 1024, 2048, 4096, 8192];  % Nystrom methods
    landmark_sizes = [200, 200, 200, 200, 200];  % m for each Nystrom training size
    
    num_sizes_pure = length(training_sizes_pure);
    num_sizes_nystrom = length(training_sizes_nystrom);
    
    % Optimization parameters
    max_iter = 100;
    tol = 1e-8;
    reps = 5; 
    maxIterKM = 200; 

    % --- 2. Initialize Result Storage (using NaN) ---
    % Epsilon values
    epsilon_opt_rippa = nan(num_sizes_pure, 1);
    epsilon_opt_gd = nan(num_sizes_pure, 1);
    epsilon_opt_rippa_pure = nan(num_sizes_pure, 1);
    epsilon_opt_gd_pure = nan(num_sizes_pure, 1);

    % Timing
    elapsed_time_rippa = nan(num_sizes_pure, 1);
    elapsed_time_gd = nan(num_sizes_pure, 1);
    elapsed_time_rippa_pure = nan(num_sizes_pure, 1);
    elapsed_time_gd_pure = nan(num_sizes_pure, 1);
    
    % Final Errors
    mean_error_gd = nan(num_sizes_pure, 1);
    mean_error_rippa = nan(num_sizes_pure, 1);
    mean_error_rippa_pure = nan(num_sizes_pure, 1);
    mean_error_gd_pure = nan(num_sizes_pure, 1);

    % --- 3. Create Fixed Test Set ---
    N_test = 5000;
    X_test = rand(N_test, 3);
    f_test = testcase(X_test);

    % --- 4. Main Loop 1: Pure Methods ---
    fprintf('\n=== Processing Pure Methods ===\n');
    
    for i = 1:num_sizes_pure
        N = training_sizes_pure(i);
        
        fprintf('\n--- Pure Methods: Training Size N = %d (%d/%d) ---\n', N, i, num_sizes_pure);

        % --- Part A: Generate Data & Optimize Epsilon ---
        X_train = rand(N, 3);
        X_train = 0.5 * (1 - cos(pi * X_train));
        f_train = testcase(X_train);
        
        % Rippa (Pure)
        fprintf('  [1/2] Running Rippa Pure...\n');
        tic;
        epsilon_opt_rippa_pure(i) = optimize_epsilon_pure(X_train, f_train);
        elapsed_time_rippa_pure(i) = toc;
        
        % GD (Pure)
        fprintf('  [2/2] Running GD Pure...\n');
        tic;
        N_sample = min(N, 1000);
        dist_sample = pdist(X_train(1:N_sample, :));
        median_dist = median(dist_sample);
        epsilon_init_pure = 1.0 / (median_dist + 1e-8);
        epsilon_opt_gd_pure(i) = optimize_epsilon_gd_pure(X_train, f_train, epsilon_init_pure, max_iter, tol);
        elapsed_time_gd_pure(i) = toc;
        
        fprintf('  Optimized Epsilons (GD_Pure/Rippa_Pure): %.2e / %.2e\n', ...
            epsilon_opt_gd_pure(i), epsilon_opt_rippa_pure(i));

        % --- Part B: Compute Lambda Coefficients ---
        
        % Rippa (Pure) Lambda
        A_train_rippa_pure = imq_rbf(pdist2(X_train, X_train), epsilon_opt_rippa_pure(i));
        A_train_rippa_pure = A_train_rippa_pure + 1e-10 * eye(N);
        lambda_train_rippa_pure = A_train_rippa_pure \ f_train;
        
        % GD (Pure) Lambda
        A_train_gd_pure = imq_rbf(pdist2(X_train, X_train), epsilon_opt_gd_pure(i));
        A_train_gd_pure = A_train_gd_pure + 1e-10 * eye(N);
        lambda_train_gd_pure = A_train_gd_pure \ f_train;
        
        % --- Part C: Evaluate Error on Test Set ---
        
        % Rippa (Pure) Error
        A_test_rippa_pure = imq_rbf(pdist2(X_test, X_train), epsilon_opt_rippa_pure(i));
        f_pred_rippa_pure = A_test_rippa_pure * lambda_train_rippa_pure;
        mean_error_rippa_pure(i) = sqrt(norm(f_pred_rippa_pure - f_test)^2 / N_test);
        
        % GD (Pure) Error
        A_test_gd_pure = imq_rbf(pdist2(X_test, X_train), epsilon_opt_gd_pure(i));
        f_pred_gd_pure = A_test_gd_pure * lambda_train_gd_pure;
        mean_error_gd_pure(i) = sqrt(norm(f_pred_gd_pure - f_test)^2 / N_test);
        
        fprintf('  Mean RMSE (GD_Pure/Rippa_Pure): %.6e / %.6e\n', ...
            mean_error_gd_pure(i), mean_error_rippa_pure(i));
            
        fprintf('--- Completed Pure Methods, Size %d ---\n', N);
    end
    
    % --- 5. Main Loop 2: Nystrom Methods ---
    fprintf('\n=== Processing Nystrom Methods ===\n');
    
    for i = 1:num_sizes_nystrom
        N = training_sizes_nystrom(i);
        m = landmark_sizes(i);
        
        % Find corresponding pure index
        pure_idx = find(training_sizes_pure == N);
        
        fprintf('\n--- Nystrom Methods: Training Size N = %d, Landmarks m = %d (%d/%d) ---\n', ...
            N, m, i, num_sizes_nystrom);

        % --- Part A: Generate Data & Optimize Epsilon ---
        X_train = rand(N, 3);
        X_train = 0.5 * (1 - cos(pi * X_train));
        f_train = testcase(X_train);
        
        % Rippa (Nystrom)
        fprintf('  [1/2] Running Rippa + Nystrom...\n');
        tic;
        landmark_indices_rippa = kmeans_landmarks_nn(X_train, m, reps, maxIterKM);
        epsilon_opt_rippa(pure_idx) = optimize_epsilon(X_train, f_train, landmark_indices_rippa);
        elapsed_time_rippa(pure_idx) = toc;
        
        % GD (Nystrom)
        fprintf('  [2/2] Running GD + Nystrom...\n');
        tic;
        N_sample = min(N, 1000);
        dist_sample = pdist(X_train(1:N_sample, :));
        median_dist = median(dist_sample);
        epsilon_init = 1.0 / (median_dist + 1e-8);
        landmark_indices_gd = kmeans_landmarks_nn(X_train, m, reps, maxIterKM);
        epsilon_opt_gd(pure_idx) = optimize_epsilon_gd(X_train, f_train, epsilon_init, max_iter, tol, landmark_indices_gd);
        elapsed_time_gd(pure_idx) = toc;
        
        fprintf('  Optimized Epsilons (GD+Nystrom/Rippa+Nystrom): %.2e / %.2e\n', ...
            epsilon_opt_gd(pure_idx), epsilon_opt_rippa(pure_idx));

        % --- Part B: Compute Lambda Coefficients ---
        
        % GD (Nystrom) Lambda
        A_train_gd = imq_rbf(pdist2(X_train, X_train), epsilon_opt_gd(pure_idx));
        A_train_gd = A_train_gd + 1e-10 * eye(N);
        lambda_train_gd = A_train_gd \ f_train;
        
        % Rippa (Nystrom) Lambda
        A_train_rippa = imq_rbf(pdist2(X_train, X_train), epsilon_opt_rippa(pure_idx));
        A_train_rippa = A_train_rippa + 1e-10 * eye(N);
        lambda_train_rippa = A_train_rippa \ f_train;
        
        % Recalculate Pure methods Lambda with same data
        A_train_rippa_pure = imq_rbf(pdist2(X_train, X_train), epsilon_opt_rippa_pure(pure_idx));
        A_train_rippa_pure = A_train_rippa_pure + 1e-10 * eye(N);
        lambda_train_rippa_pure = A_train_rippa_pure \ f_train;
        
        A_train_gd_pure = imq_rbf(pdist2(X_train, X_train), epsilon_opt_gd_pure(pure_idx));
        A_train_gd_pure = A_train_gd_pure + 1e-10 * eye(N);
        lambda_train_gd_pure = A_train_gd_pure \ f_train;
        
        % --- Part C: Evaluate Error on Test Set ---
        
        % GD (Nystrom) Error
        A_test_gd = imq_rbf(pdist2(X_test, X_train), epsilon_opt_gd(pure_idx));
        f_pred_gd = A_test_gd * lambda_train_gd;
        mean_error_gd(pure_idx) = sqrt(norm(f_pred_gd - f_test)^2 / N_test);
        
        % Rippa (Nystrom) Error
        A_test_rippa = imq_rbf(pdist2(X_test, X_train), epsilon_opt_rippa(pure_idx));
        f_pred_rippa = A_test_rippa * lambda_train_rippa;
        mean_error_rippa(pure_idx) = sqrt(norm(f_pred_rippa - f_test)^2 / N_test);
        
        % Rippa (Pure) Error (recalculated)
        A_test_rippa_pure = imq_rbf(pdist2(X_test, X_train), epsilon_opt_rippa_pure(pure_idx));
        f_pred_rippa_pure = A_test_rippa_pure * lambda_train_rippa_pure;
        mean_error_rippa_pure(pure_idx) = sqrt(norm(f_pred_rippa_pure - f_test)^2 / N_test);
        
        % GD (Pure) Error (recalculated)
        A_test_gd_pure = imq_rbf(pdist2(X_test, X_train), epsilon_opt_gd_pure(pure_idx));
        f_pred_gd_pure = A_test_gd_pure * lambda_train_gd_pure;
        mean_error_gd_pure(pure_idx) = sqrt(norm(f_pred_gd_pure - f_test)^2 / N_test);
        
        fprintf('  Mean RMSE (GD+Nystrom/Rippa+Nystrom/GD_Pure/Rippa_Pure): %.6e / %.6e / %.6e / %.6e\n', ...
            mean_error_gd(pure_idx), mean_error_rippa(pure_idx), mean_error_gd_pure(pure_idx), mean_error_rippa_pure(pure_idx));
            
        fprintf('--- Completed Nystrom Methods, Size %d ---\n', N);
    end
    
    fprintf('\n--- All runs complete. Plotting results. ---\n');

    % --- 6. Plot Results ---
    figure('Position', [100, 100, 1200, 600]);

    % Pure Rippa - all points
    semilogy(training_sizes_pure, mean_error_rippa_pure, '^:', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.9290 0.6940 0.1250], 'DisplayName', 'Rippa');
    hold on;

    % Pure GD - all points
    semilogy(training_sizes_pure, mean_error_gd_pure, 'd-.', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD');

    % Rippa + Nystrom - filter NaN
    valid_rippa = ~isnan(mean_error_rippa);
    semilogy(training_sizes_pure(valid_rippa), mean_error_rippa(valid_rippa), ...
        's--', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Rippa + Nystrom');

    % GD + Nystrom - filter NaN
    valid_gd = ~isnan(mean_error_gd);
    semilogy(training_sizes_pure(valid_gd), mean_error_gd(valid_gd), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');

    % Labels and title
    xlabel('Training Size (N)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Root Mean Squared Error (RMSE)', 'FontSize', 14, 'FontWeight', 'bold');
    title('RMSE Comparison of Epsilon Optimization Methods (3D)', 'FontSize', 16, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 12);
    grid on;
    box on;

    set(gca, 'XScale', 'log');
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 12);
    xlim([50, 10000]);
    hold off;

    fprintf('--- Complete! ---\n');
end
% -------------------------------------------------------------------
%  [KEEP ALL OTHER HELPER FUNCTIONS AS THEY ARE]
%  (testcase, calculate_rmse, gaussian_rbf, kmeans_landmarks_nn,
%   compute_loo_error_norm, line_search, compute_loo_gradient, 
%   optimize_epsilon_gd, optimize_epsilon, ...and all PURE functions...)
% -------------------------------------------------------------------


function f = testcase(X)
    % 3D Test Function
    % Using a simple smooth function for demonstration
    x = X(:,1); 
    y = X(:,2);
    z = X(:,3);
%   f = ones(size(X, 1), 1);
   %   f = sin(x.^2 + 2.* y.^2) - sin(2.* x.^2 + (y-0.5).^2 + z.^2);
     f = sin(2 * pi .*(x.^2 + 2 .* y.^2)) -sin(2 *pi .* (2.* x.^2 +(y-0.5).^2 + z.^2));

end

function rmse = calculate_rmse(X_train, f_train, X_test, f_test, epsilon,N)
    % Calculates RMSE on the test set for a given trained model
    
    % 1. Solve for lambda coefficients (interpolation)
    % Using the "Pure" method for final interpolation as it's more stable
   % N = size(X_train, 1);
    A_train = imq_rbf(pdist2(X_train, X_train), epsilon);
    A_train_reg = A_train + 1e-14 * eye(N);
    
    try
        lambda_train = A_train_reg \ f_train;
    catch
        fprintf('Warning: Matrix is singular, using pinv for RMSE calculation.\n');
        lambda_train = pinv(A_train_reg) * f_train;
    end
    
    % 2. Predict on test set
    A_test = imq_rbf(pdist2(X_test, X_train), epsilon);
    f_pred = A_test * lambda_train;
    
    % 3. Calculate RMSE
    rmse = sqrt(mean((f_pred - f_test).^2)/N);
end


function phi = imq_rbf(r, epsilon)
    % Inverse Multiquadratic RBF (IMQ)
    % Standard form: (1 + (epsilon*r)^2)^(-1/2)
    phi = 1 ./ sqrt(1 + (epsilon * r).^2);
end
function landmark_indices = kmeans_landmarks_nn(X, m, reps, maxIter)
    if nargin < 3, reps = 5; end
    if nargin < 4, maxIter = 200; end


    [labels, C] = kmeans(X, m, ...
        'Replicates', reps, ...
        'MaxIter', maxIter, ...
        'EmptyAction', 'singleton', ...
        'OnlinePhase', 'on', ...
        'Start', 'plus'); 
    
    landmark_indices = knnsearch(X, C); 

    if numel(unique(landmark_indices)) < m
        D = pdist2(C, X);
        taken = false(size(X,1),1);
        landmark_indices = zeros(1,m);
        for j = 1:m
            [~, order] = sort(D(j,:),'ascend');
            kptr = 1;
            while taken(order(kptr)), kptr = kptr + 1; end
            pick = order(kptr);
            landmark_indices(j) = pick;
            taken(pick) = true;
        end
    end
end
function landmark_indices = kmeans_landmarks_nn_old(X, m, reps, maxIter)
    % Selects m landmarks from X using k-means
    if nargin < 3, reps = 5; end
    if nargin < 4, maxIter = 200; end

    [~, C] = kmeans(X, m, ...
        'Replicates', reps, ...
        'MaxIter', maxIter, ...
        'EmptyAction', 'singleton', ...
        'OnlinePhase', 'on'); 

    landmark_indices = knnsearch(X, C); 
    
    % Handle rare case of duplicate indices
    if numel(unique(landmark_indices)) < m
        D = pdist2(C, X);
        taken = false(size(X,1),1);
        landmark_indices = zeros(m,1); % Ensure column vector
        for j = 1:m
            [~, order] = sort(D(j,:),'ascend');
            kptr = 1;
            while kptr <= length(order) && taken(order(kptr))
                kptr = kptr + 1;
            end
            if kptr <= length(order)
                pick = order(kptr);
                landmark_indices(j) = pick;
                taken(pick) = true;
            else
                % Fallback: find first untaken point
                fallback_idx = find(~taken, 1);
                if ~isempty(fallback_idx)
                    landmark_indices(j) = fallback_idx;
                    taken(fallback_idx) = true;
                else
                    fprintf('Warning: Could not find %d unique landmarks.\n', m);
                    break; 
                end
            end
        end
        % Ensure unique indices, if k-means failed badly
        landmark_indices = unique(landmark_indices, 'stable');
        if length(landmark_indices) < m
             fprintf('Warning: Only found %d unique landmarks for m=%d.\n', length(landmark_indices), m);
             % If still not enough, just take the first m points
             all_indices = (1:size(X,1))';
             untaken = setdiff(all_indices, landmark_indices);
             needed = m - length(landmark_indices);
             if length(untaken) >= needed
                landmark_indices = [landmark_indices; untaken(1:needed)];
             else % This should almost never happen
                 landmark_indices = [landmark_indices; untaken(1:end)];
                 while length(landmark_indices) < m
                     landmark_indices(end+1) = landmark_indices(1); % Just duplicate
                 end
             end
        end
        landmark_indices = landmark_indices(1:m); % Force size m
    end
    landmark_indices = landmark_indices(:); % Ensure column vector
end


% -------------------------------------------------------------------
%  METHOD 1 & 2: NYSTROM-WOODBURY (m < N)
% -------------------------------------------------------------------

function L = compute_loo_error_norm(epsilon, X, landmark_indices, f)
    % Woodbury-based LOOCV (Dimension-Agnostic)
    % A = lambda*I + C*W_inv*C'
    % A_inv = lambda_inv*I - lambda_inv^2 * C * (W + lambda_inv*C'C)_inv * C'

    lambda = 1e-6; % Regularization (as used in 2D)
    N = size(X, 1);
    m = length(landmark_indices);

    % Handle N=m case (no approximation)
    if N == m
        % Fallback to "Pure" method
        L = compute_loo_error_norm_pure(epsilon, X, f);
        return;
    end

    % 1) Build C, W directly
    Xl = X(landmark_indices, :);
    C  = imq_rbf(pdist2(X,  Xl),  epsilon);   % N×m
    W  = imq_rbf(pdist2(Xl, Xl),  epsilon);   % m×m

    % 2) Regularize W
    W = W + 1e-12*eye(m);

    % 3) Form M = W + (1/lambda) * C'C   (m×m)
    CtC = C.' * C;                 % m×m
    M   = W + (1/lambda) * CtC;    % m×m

    % 4) Factorize M (Cholesky preferred, SVD fallback)
    try
        [R,p] = chol(M);
        if p ~= 0, error('M not positive definite'); end % Throw to catch block
        
        % 4a) Use triangular solve
        Cf      = C.' * f;               % m×1
        MinvCf  = R \ (R.' \ Cf);        % m×1

        % 4b) Calculate A_inv * f
        Ainv_f  = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);   % N×1

        % 4c) Compute Minv for calculating diagonal
        Im = eye(m);
        Minv = R \ (R.' \ Im);           % m×m
    catch
        % Fallback to SVD
        % fprintf('Warning: Cholesky failed for M (eps=%.2e). Using SVD.\n', epsilon);
        [U,S,V] = svd(M);
        s  = diag(S);
        s_inv = 1 ./ max(s, 1e-15);
        Minv = V * diag(s_inv) * U.'; % M_inv = V*S_inv*U'
        
        Cf     = C.' * f;
        MinvCf = Minv * Cf;
        Ainv_f = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);
    end

    % 5) diag(A_inv) = lambda_inv - lambda_inv^2 * rowdot( C*Minv , C )
    T = C * Minv;                                      % N×m
    diagAinv = (1/lambda) - (1/lambda^2) * sum(T .* C, 2);  % N×1
    diagAinv = max(diagAinv, 1e-15); % Avoid division by zero

    % 6) Rippa's LOO error vector and objective
    E = Ainv_f ./ diagAinv;                            % N×1
    L = norm(E)^2;
    
    if isnan(L) || isinf(L)
        fprintf('Warning: L is NaN/Inf (eps=%.2e). Returning large value.\n', epsilon);
       L = 1e100; % Return large value if unstable
    end
end

function [eta, L_new] = line_search(epsilon, grad, X, f, landmark_indices, eta_ini, gamma, alpha, max_iters, min_eta)
    % Backtracking line search for Nystrom method
    if nargin < 6, eta_ini = 1.0; end
    if nargin < 7, gamma = 0.01; end
    if nargin < 8, alpha = 0.5; end
    if nargin < 9, max_iters = 50; end
    if nargin < 10, min_eta = 1e-12; end
    
    eta = eta_ini;
    L_current = compute_loo_error_norm(epsilon, X, landmark_indices,f);
    
    if isnan(L_current) || isinf(L_current)
        L_new = L_current;
        eta = min_eta;
        return; % Cannot perform line search from NaN/Inf
    end
    
    for iter = 1:max_iters
        epsilon_new = max(epsilon - eta * grad, 1e-8);  % Epsilon must be positive
        
        if abs(epsilon_new - epsilon) < 1e-15
             L_new = L_current; % Avoid re-computation if step is too small
             break;
        end
        
        L_new = compute_loo_error_norm(epsilon_new, X, landmark_indices, f);
        
        if isnan(L_new) || isinf(L_new)
            eta = eta * alpha; % Instability, reduce step
            continue;
        end
        
        % Armijo condition
        if L_new <= L_current - gamma * eta * grad^2
            break; % Step accepted
        end
        
        eta = eta * alpha;
        
        if eta < min_eta
            eta = min_eta;
            L_new = compute_loo_error_norm(max(epsilon - eta * grad, 1e-8), X, landmark_indices, f);
            if isnan(L_new) || isinf(L_new) % Check final step
                L_new = L_current;
                eta = 0;
            end
            break;
        end
    end
    if isnan(L_new) % If loop ended on a NaN, revert
        L_new = L_current;
        eta = 0;
    end
end

function [L, grad] = compute_loo_gradient(epsilon, X, f, landmark_indices)
    % Numerical gradient for Nystrom method
    h = max(1e-8, 1e-8 * abs(epsilon)); 
    
    L_plus = compute_loo_error_norm(epsilon + h, X, landmark_indices,f);
    L_minus = compute_loo_error_norm(epsilon - h, X, landmark_indices,f);
    
    grad = (L_plus - L_minus) / (2*h);
    
    if isnan(grad) || abs(grad) > 1e20 % Handle extreme gradients
        grad = 1e20 * sign(grad);
        if grad < 0
             grad = -1e20;
        else
             grad = 1e20;
        end
    end
    
    L = compute_loo_error_norm(epsilon, X, landmark_indices, f); % Use center point
    if isnan(L) || isinf(L)
        L = L_plus; % Try to get a non-nan value
    end
    if isnan(L) || isinf(L)
        L = L_minus;
    end
end

function [epsilon_opt, history,iter] = optimize_epsilon_gd_old(X, f, epsilon_init, max_iter, tol, landmark_indices)
    % GD for Nystrom method
    epsilon = max(epsilon_init, 1e-8);  
    history.epsilon = zeros(max_iter + 1, 1); % Fix indexing
    history.loss = inf(max_iter + 1, 1);
    
    best_epsilon = epsilon;
    best_loss = compute_loo_error_norm(epsilon, X, landmark_indices, f);
    if isnan(best_loss) || isinf(best_loss)
        fprintf('Error: Initial epsilon for GD-Nystrom is unstable. Aborting.\n');
        epsilon_opt = epsilon_init;
        history.epsilon = epsilon;
        history.loss = best_loss;
        return;
    end
    history.loss(1) = best_loss;
    history.epsilon(1) = epsilon;
    
    iter = 1;
    while iter <= max_iter
        [L, grad] = compute_loo_gradient(epsilon, X, f, landmark_indices);
        
        if isnan(L) || isnan(grad)
             fprintf('Warning: GD-Nystrom encountered NaN. Stopping at iter %d.\n', iter);
             break;
        end

        history.epsilon(iter) = epsilon;
        history.loss(iter) = L;
        
        if L < best_loss
            best_loss = L;
            best_epsilon = epsilon;
        end
        
        if abs(grad) < tol || (iter > 10 && abs(history.loss(iter) - history.loss(iter-1)) < tol*abs(history.loss(iter-1)))
            % fprintf('GD-Nystrom converged at iter %d.\n', iter);
            break;
        end
        
        [eta, L_new] = line_search(epsilon, grad, X, f, landmark_indices);
        
        if eta == 0 || isnan(L_new) || isinf(L_new)
             % fprintf('Line search failed or returned NaN/Inf. Stopping at iter %d.\n', iter);
             break; % Line search failed
        end
        
        epsilon_new = max(epsilon - eta * grad, 1e-8);
        
        if abs(epsilon_new - epsilon) < 1e-12
            % fprintf('Step size too small. Stopping at iter %d.\n', iter);
            break; % Step too small
        end
        
        epsilon = epsilon_new;
        iter = iter + 1; % Increment iter
        history.epsilon(iter) = epsilon; % Store final state
        history.loss(iter) = L_new;
 

    end

    % Final check on best loss
    if iter > 0 && history.loss(iter) > best_loss
        epsilon_opt = best_epsilon;
         fprintf('GD-Nystrom: Using best solution (L=%.2e) instead of final (L=%.2e).\n', best_loss, history.loss(iter));
    display(iter);
    
    else
        epsilon_opt = epsilon;
    end
    
    history.epsilon = history.epsilon(1:iter);
    history.loss = history.loss(1:iter);
end
function [epsilon_opt, history, iter] = optimize_epsilon_gd(X, f, epsilon_init, max_iter, tol, landmark_indices)
    % GD for Nystrom method (Log-space optimization version)

    epsilon = max(epsilon_init, 1e-8);
    log_eps = log(epsilon);
    
    history.epsilon = zeros(max_iter + 1, 1);
    history.loss = inf(max_iter + 1, 1);
    

    best_loss = compute_loo_error_norm(epsilon, X, landmark_indices, f);
    
    if isnan(best_loss) || isinf(best_loss)
        fprintf('Error: Initial epsilon for GD-Nystrom is unstable. Aborting.\n');
        epsilon_opt = epsilon_init;
        history.epsilon(1) = epsilon;
        history.loss(1) = best_loss;
        iter = 0;
        return;
    end
    
    history.loss(1) = best_loss;
    history.epsilon(1) = epsilon;
    best_log_eps = log_eps;
    best_epsilon = epsilon;
    
    eta = 1.0; 
    

    for iter = 1:max_iter
     
        epsilon = exp(log_eps);
        
        
        [L, grad] = compute_loo_gradient(epsilon, X, f, landmark_indices);
        
        if isnan(L) || isnan(grad)
             fprintf('Warning: GD-Nystrom encountered NaN. Stopping at iter %d.\n', iter);
             break;
        end
        
       
        history.epsilon(iter + 1) = epsilon;
        history.loss(iter + 1) = L;
        
        
        if L < best_loss
            best_loss = L;
            best_epsilon = epsilon;
            best_log_eps = log_eps;
        end
        
      
        % dL/d(log_eps) = dL/d(eps) * d(eps)/d(log_eps) = grad * epsilon
        grad_log = grad * epsilon;
        
     
        if abs(grad_log) < tol
            fprintf('Converged: Gradient magnitude small (Iter %d)\n', iter);
            break;
        end
        if iter > 10 && abs(history.loss(iter+1) - history.loss(iter)) < 1e-12 * abs(history.loss(iter))
            % fprintf('Converged: Loss stabilized at iter %d.\n', iter);
            break;
        end
        
   
        current_eta = eta;
        accept_step = false;
        
        
        for search_iter = 1:15
            log_eps_try = log_eps - current_eta * grad_log;
            epsilon_try = exp(log_eps_try);
            
            
            L_new = compute_loo_error_norm(epsilon_try, X, landmark_indices, f);
            
            if isnan(L_new) || isinf(L_new)
                current_eta = current_eta * 0.5;
                continue;
            end
            
           
            if L_new < L
                log_eps = log_eps_try; 
                eta = min(current_eta * 1.2, 5.0); 
                accept_step = true;
                break;
            else
                current_eta = current_eta * 0.5; 
            end
        end
        
        if ~accept_step
            fprintf('Line search failed. Stopping at iter %d.\n', iter);
            break;
        end
        
        
        if abs(log_eps - best_log_eps) < 1e-12 && iter > 5
             % break; 
        end
    end
    
  
    if iter > 0 && history.loss(iter+1) > best_loss
        epsilon_opt = best_epsilon;
        fprintf('GD-Nystrom: Using best solution (L=%.4e) instead of final (L=%.4e).\n', best_loss, history.loss(iter+1));
    else
        epsilon_opt = exp(log_eps);
    end
    
  
    history.epsilon = history.epsilon(1:iter+1);
    history.loss = history.loss(1:iter+1);
    
   
end

function [optimal_epsilon, min_E_norm] = optimize_epsilon(X, f, landmark_indices)
   
    num_coarse = 30; 
    grid_coarse = logspace(log10(1e-5), log10(1000), num_coarse);
    
    E_norms_coarse = zeros(num_coarse, 1);
    
    % fprintf('  > Stage 1: Coarse search over %d points...\n', num_coarse);
    for idx = 1:num_coarse
        E_norms_coarse(idx) = compute_loo_error_norm(grid_coarse(idx), X, landmark_indices, f);
    end
    
   
    [min_val_coarse, min_idx_coarse] = min(E_norms_coarse);
    
    
    if isinf(min_val_coarse) || isnan(min_val_coarse)
        fprintf('Warning: Rippa-Nystrom (Coarse) failed. Validating...\n');
        valid_mask = ~isnan(E_norms_coarse) & ~isinf(E_norms_coarse);
        if any(valid_mask)
            [min_val_coarse, valid_idx] = min(E_norms_coarse(valid_mask));
            full_indices = find(valid_mask);
            min_idx_coarse = full_indices(valid_idx);
        else
          
            optimal_epsilon = 1.0;
            min_E_norm = 1e100;
            return;
        end
    end
    
    best_coarse_epsilon = grid_coarse(min_idx_coarse);
    % fprintf('    Best coarse epsilon: %.4f (Error: %.4e)\n', best_coarse_epsilon, min_val_coarse);

   
    
    lower_bound = best_coarse_epsilon * 0.5;
    upper_bound = best_coarse_epsilon * 2.0;
    
    
    if min_idx_coarse == 1
        lower_bound = best_coarse_epsilon * 0.1; 
    elseif min_idx_coarse == num_coarse
        upper_bound = best_coarse_epsilon * 10.0; 
    end
    
    num_fine = 50; 
    grid_fine = linspace(lower_bound, upper_bound, num_fine);
    
    E_norms_fine = zeros(num_fine, 1);
    
    % fprintf('  > Stage 2: Fine search over %d points around %.4f...\n', num_fine, best_coarse_epsilon);
    for idx = 1:num_fine
        E_norms_fine(idx) = compute_loo_error_norm(grid_fine(idx), X, landmark_indices, f);
    end
    
    
    [min_val_fine, min_idx_fine] = min(E_norms_fine);
    
  
    if min_val_fine < min_val_coarse
        min_E_norm = min_val_fine;
        optimal_epsilon = grid_fine(min_idx_fine);
    else
        min_E_norm = min_val_coarse;
        optimal_epsilon = best_coarse_epsilon;
    end
    
    % fprintf('    Final Optimal Epsilon: %.4f\n', optimal_epsilon);
end


% -------------------------------------------------------------------
%  METHOD 3 & 4: PURE (N x N)
% -------------------------------------------------------------------

function L = compute_loo_error_norm_pure(epsilon, X, f)
    % Standard O(N^3) LOOCV
    N = size(X, 1);
    r_dist = pdist2(X, X);
    A = imq_rbf(r_dist, epsilon);
    
    lambda_reg = 1e-12 * norm(A, 'fro');
    A_reg = A + lambda_reg * eye(N);
    
    try
        [R, p] = chol(A_reg);
        if p ~= 0, error('M not positive definite'); end
        
        invR = inv(R); % N x N
        invDiag = sum(invR.^2, 2);  % diag(inv(A)) = sum(inv(R).^2, 2)
        lambda_coeff = R \ (R' \ f);
    catch
        % Fallback to SVD
        % fprintf('Warning: Cholesky failed for Pure (eps=%.2e). Using SVD.\n', epsilon);
        [U, S, V] = svd(A_reg);
        s = diag(S);
        s_inv = 1 ./ max(s, 1e-15);
        
        invDiag = sum((V .* s_inv').^2, 2); % s_inv' is 1xN
        lambda_coeff = V * (diag(s_inv) * (U' * f));
    end
    
    E = lambda_coeff ./ max(invDiag, 1e-15);
    L = norm(E)^2;
    
    if isnan(L) || isinf(L)
       % fprintf('Warning: Pure L is NaN/Inf (eps=%.2e). Returning large value.\n', epsilon);
       L = 1e100; % Return large value if unstable
    end
end

function [eta, L_new] = line_search_pure(epsilon, grad, X, f, eta_ini, gamma, alpha, max_iters, min_eta)
    % Backtracking line search for Pure method
    if nargin < 5, eta_ini = 1.0; end
    if nargin < 6, gamma = 0.01; end
    if nargin < 7, alpha = 0.5; end
    if nargin < 8, max_iters = 50; end
    if nargin < 9, min_eta = 1e-12; end
    
    eta = eta_ini;
    L_current = compute_loo_error_norm_pure(epsilon, X, f);

    if isnan(L_current) || isinf(L_current)
        L_new = L_current;
        eta = min_eta;
        return; 
    end
    
    for iter = 1:max_iters
        epsilon_new = max(epsilon - eta * grad, 1e-8);
        
        if abs(epsilon_new - epsilon) < 1e-15
             L_new = L_current;
             break;
        end
        
        L_new = compute_loo_error_norm_pure(epsilon_new, X, f);
        
        if isnan(L_new) || isinf(L_new)
            eta = eta * alpha;
            continue;
        end
        
        if L_new <= L_current - gamma * eta * grad^2
            break;
        end
        
        eta = eta * alpha;
        
        if eta < min_eta
            eta = min_eta;
            L_new = compute_loo_error_norm_pure(max(epsilon - eta * grad, 1e-8), X, f);
            if isnan(L_new) || isinf(L_new)
                L_new = L_current;
                eta = 0;
            end
            break;
        end
    end
    if isnan(L_new)
        L_new = L_current;
        eta = 0;
    end
end

function [L, grad] = compute_loo_gradient_pure(epsilon, X, f)
    % Numerical gradient for Pure method
    h = max(1e-8, 1e-8 * abs(epsilon));
    
    L_plus = compute_loo_error_norm_pure(epsilon + h, X, f);
    L_minus = compute_loo_error_norm_pure(epsilon - h, X, f);
    
    grad = (L_plus - L_minus) / (2*h);
    
    if isnan(grad) || abs(grad) > 1e20
        grad = 1e20 * sign(grad);
         if grad < 0
             grad = -1e20;
        else
             grad = 1e20;
        end
    end
    
    L = compute_loo_error_norm_pure(epsilon, X, f); % Use center point
    if isnan(L) || isinf(L)
        L = L_plus;
    end
    if isnan(L) || isinf(L)
        L = L_minus;
    end
end

function [epsilon_opt, history] = optimize_epsilon_gd_pure_old(X, f, epsilon_init, max_iter, tol)
    % GD for Pure method
    epsilon = max(epsilon_init, 1e-8);
    history.epsilon = zeros(max_iter + 1, 1); % Fix indexing
    history.loss = inf(max_iter + 1, 1);
    
    best_epsilon = epsilon;
    best_loss = compute_loo_error_norm_pure(epsilon, X, f);
    if isnan(best_loss) || isinf(best_loss)
        fprintf('Error: Initial epsilon for GD-Pure is unstable. Aborting.\n');
        epsilon_opt = epsilon_init;
        history.epsilon = epsilon;
        history.loss = best_loss;
        return;
    end
    history.loss(1) = best_loss;
    history.epsilon(1) = epsilon;

    iter = 1;
    while iter <= max_iter
        [L, grad] = compute_loo_gradient_pure(epsilon, X, f);
        
        if isnan(L) || isnan(grad)
             fprintf('Warning: GD-Pure encountered NaN. Stopping at iter %d.\n', iter);
             break;
        end

        history.epsilon(iter) = epsilon;
        history.loss(iter) = L;
        
        if L < best_loss
            best_loss = L;
            best_epsilon = epsilon;
        end
        
        if abs(grad) < tol || (iter > 10 && abs(history.loss(iter) - history.loss(iter-1)) < tol*abs(history.loss(iter-1)))
            % fprintf('GD-Pure converged at iter %d.\n', iter);
            break;
        end
        
        [eta, L_new] = line_search_pure(epsilon, grad, X, f);
        
        if eta == 0 || isnan(L_new) || isinf(L_new)
            % fprintf('Line search failed or returned NaN/Inf. Stopping at iter %d.\n', iter);
            break;
        end
        
        epsilon_new = max(epsilon - eta * grad, 1e-8);
        
        if abs(epsilon_new - epsilon) < 1e-12
            % fprintf('Step size too small. Stopping at iter %d.\n', iter);
            break;
        end
        
        epsilon = epsilon_new;
        iter = iter + 1; % Increment iter
        history.epsilon(iter) = epsilon; % Store final state
        history.loss(iter) = L_new;
    end
    
    if iter > 0 && history.loss(iter) > best_loss
        epsilon_opt = best_epsilon;
        % fprintf('GD-Pure: Using best solution (L=%.2e) instead of final (L=%.2e).\n', best_loss, history.loss(iter));
    else
        epsilon_opt = epsilon;
    end
    
    history.epsilon = history.epsilon(1:iter);
    history.loss = history.loss(1:iter);
end

function [epsilon_opt, history] = optimize_epsilon_gd_pure(X, f, epsilon_init, max_iter, tol)
    % GD for Pure method (Log-space optimization version)
   
    
    
    epsilon = max(epsilon_init, 1e-8);
    log_eps = log(epsilon);
    
    history.epsilon = zeros(max_iter + 1, 1);
    history.loss = inf(max_iter + 1, 1);
    
   
    best_loss = compute_loo_error_norm_pure(epsilon, X, f);
    
    if isnan(best_loss) || isinf(best_loss)
        fprintf('Error: Initial epsilon for GD-Pure is unstable. Aborting.\n');
        epsilon_opt = epsilon_init;
        history.epsilon(1) = epsilon;
        history.loss(1) = best_loss;
        return;
    end
    
    history.loss(1) = best_loss;
    history.epsilon(1) = epsilon;
    
    best_epsilon = epsilon;
    best_log_eps = log_eps;
    
    eta = 1.0; 
    
    
    for iter = 1:max_iter
        
        epsilon = exp(log_eps);
        
        
        [L, grad] = compute_loo_gradient_pure(epsilon, X, f);
        
        if isnan(L) || isnan(grad)
             fprintf('Warning: GD-Pure encountered NaN. Stopping at iter %d.\n', iter);
             break;
        end
        
        
        history.epsilon(iter + 1) = epsilon;
        history.loss(iter + 1) = L;
        
        
        if L < best_loss
            best_loss = L;
            best_epsilon = epsilon;
            best_log_eps = log_eps;
        end
        
       
        % dL/d(log_eps) = grad * epsilon
        grad_log = grad * epsilon;
        
        
        if abs(grad_log) < tol
            % fprintf('GD-Pure Converged: Gradient magnitude small (Iter %d)\n', iter);
            break;
        end
        if iter > 10 && abs(history.loss(iter+1) - history.loss(iter)) < 1e-12 * abs(history.loss(iter))
            break;
        end
        
       
        current_eta = eta;
        accept_step = false;
        
        for search_iter = 1:15
            log_eps_try = log_eps - current_eta * grad_log;
            epsilon_try = exp(log_eps_try);
            
            
            L_new = compute_loo_error_norm_pure(epsilon_try, X, f);
            
            if isnan(L_new) || isinf(L_new)
                current_eta = current_eta * 0.5;
                continue;
            end
            
           
            if L_new < L
                log_eps = log_eps_try;
                eta = min(current_eta * 1.2, 5.0); 
                accept_step = true;
                break;
            else
                current_eta = current_eta * 0.5; 
            end
        end
        
        if ~accept_step
            % fprintf('GD-Pure Line search failed. Stopping at iter %d.\n', iter);
            break;
        end
    end
    
   
    if iter > 0 && history.loss(iter+1) > best_loss
        epsilon_opt = best_epsilon;
        fprintf('GD-Pure: Using best solution (L=%.4e) instead of final.\n', best_loss);
    else
        epsilon_opt = exp(log_eps);
    end
    
    history.epsilon = history.epsilon(1:iter+1);
    history.loss = history.loss(1:iter+1);
end



function [optimal_epsilon, min_E_norm] = optimize_epsilon_pure(X, f)
    % optimize_epsilon_pure: Two-Stage Search (Coarse-to-Fine)
    
    num_coarse = 30; 
    grid_coarse = logspace(log10(1e-5), log10(1000), num_coarse);
    
    E_norms_coarse = zeros(num_coarse, 1);
    
    for idx = 1:num_coarse
        E_norms_coarse(idx) = compute_loo_error_norm_pure(grid_coarse(idx), X, f);
    end
    
    [min_val_coarse, min_idx_coarse] = min(E_norms_coarse);
    
   
    if isinf(min_val_coarse) || isnan(min_val_coarse)
        fprintf('Warning: Rippa-Pure (Coarse) failed. Validating...\n');
        valid_mask = ~isnan(E_norms_coarse) & ~isinf(E_norms_coarse);
        if any(valid_mask)
            [min_val_coarse, valid_idx] = min(E_norms_coarse(valid_mask));
            full_indices = find(valid_mask);
            min_idx_coarse = full_indices(valid_idx);
        else
            optimal_epsilon = 1.0;
            min_E_norm = 1e100;
            return;
        end
    end
    
    best_coarse_epsilon = grid_coarse(min_idx_coarse);

    
    lower_bound = best_coarse_epsilon * 0.5;
    upper_bound = best_coarse_epsilon * 2.0;
    
   
    if min_idx_coarse == 1
        lower_bound = best_coarse_epsilon * 0.1;
    elseif min_idx_coarse == num_coarse
        upper_bound = best_coarse_epsilon * 10.0;
    end
    
    num_fine = 50; 
    grid_fine = linspace(lower_bound, upper_bound, num_fine);
    
    E_norms_fine = zeros(num_fine, 1);
    
    for idx = 1:num_fine
        E_norms_fine(idx) = compute_loo_error_norm_pure(grid_fine(idx), X, f);
    end
    
    [min_val_fine, min_idx_fine] = min(E_norms_fine);
    
   
    if min_val_fine < min_val_coarse
        min_E_norm = min_val_fine;
        optimal_epsilon = grid_fine(min_idx_fine);
    else
        min_E_norm = min_val_coarse;
        optimal_epsilon = best_coarse_epsilon;
    end
    
    % fprintf('    Pure Optimal Epsilon: %.4f\n', optimal_epsilon);
end