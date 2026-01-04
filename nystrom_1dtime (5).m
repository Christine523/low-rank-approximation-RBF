  function main()

    rng(0);
    
    
    training_sizes_pure = [64, 128, 256, 512, 1024, 2048, 4096, 8192]; 
    training_sizes_nystrom = [512, 1024, 2048, 4096, 8192]; 
    landmark_counts = [200, 200, 200, 200, 200]; 
    gd_max_iters = [100, 100, 100, 100, 100]; 
    
    num_sizes_pure = length(training_sizes_pure);
    num_sizes_nystrom = length(training_sizes_nystrom);
    
   
    reps = 5; 
    maxIterKM = 200;
    tol = 1e-8;

    
    num_iter_gd = zeros(1, num_sizes_nystrom);
    num_iter_gd_pure = zeros(1, num_sizes_pure);
    
    
    total_elapsed_time = nan(1, num_sizes_pure);  % Nystrom + GD
    total_elapsed_time_rippa = nan(1, num_sizes_pure);  % Nystrom + Rippa
    total_elapsed_time_rippa_pure = nan(1, num_sizes_pure);  % Pure Rippa
    total_elapsed_time_gd_pure = nan(1, num_sizes_pure);  % Pure GD
    
    % Epsilon
    epsilon_gd = nan(1, num_sizes_pure);
    epsilon_rippa = nan(1, num_sizes_pure);
    epsilon_rippa_pure = nan(1, num_sizes_pure);
    epsilon_gd_pure = nan(1, num_sizes_pure);

    
    
    fprintf('\n=== Processing Pure Methods ===\n');
    for i = 1:num_sizes_pure
        current_size = training_sizes_pure(i);
        
        fprintf('\n--- Pure Methods: Training size N = %d ---\n', current_size);

        
        x_train = rand(current_size, 1);
        f_train = testcase1(x_train);

        % 2. Rippa Pure
        fprintf('  [1/2] Process Rippa Pure...\n');
        tic;
        epsilon_opt_rippa_pure = optimize_epsilon_pure(x_train, f_train);
        total_elapsed_time_rippa_pure(i) = toc;
        epsilon_rippa_pure(i) = epsilon_opt_rippa_pure;

        % 3. GD Pure
        fprintf('  [2/2] Process GD Pure...\n');
        tic;
        [epsilon_opt_gd_pure, ~, num_iter_pure] = optimize_epsilon_gd_pure(x_train, f_train, franke_initial_epsilon(x_train), 100, tol);
        total_elapsed_time_gd_pure(i) = toc;
        epsilon_gd_pure(i) = epsilon_opt_gd_pure;
        num_iter_gd_pure(i) = num_iter_pure;
        
        fprintf('--- Complete Pure Methods, Size %d ---\n', current_size);
    end
    
    % --- 主循环 2: Nystrom methods (较大的training sizes) ---
    fprintf('\n=== Processing Nystrom Methods ===\n');
    for i = 1:num_sizes_nystrom
        current_size = training_sizes_nystrom(i);
        current_m = landmark_counts(i);
        current_gd_max_iter = gd_max_iters(i);
        
        % 找到对应的pure index
        pure_idx = find(training_sizes_pure == current_size);
        
        fprintf('\n--- Nystrom Methods: Training size N = %d, landmarks m = %d ---\n', current_size, current_m);

        % 1. 生成数据
        x_train = rand(current_size, 1);
        f_train = testcase1(x_train);

        % 2. Rippa with Nystrom
        fprintf('  [1/2] Process Rippa w/ Nystrom...\n');
        tic;
        landmark_indices_rippa = kmeans_landmarks_nn(x_train, current_m, reps, maxIterKM);
        epsilon_opt_rippa = optimize_epsilon(x_train, f_train, landmark_indices_rippa);
        total_elapsed_time_rippa(pure_idx) = toc;
        epsilon_rippa(pure_idx) = epsilon_opt_rippa;

        % 3. GD with Nystrom
        fprintf('  [2/2] Process GD w/ Nystrom...\n');
        tic;
        landmark_indices_gd = kmeans_landmarks_nn(x_train, current_m, reps, maxIterKM);
        [epsilon_opt_gd, ~, num_iter] = optimize_epsilon_gd(x_train, f_train, franke_initial_epsilon(x_train), current_gd_max_iter, tol, landmark_indices_gd);
        total_elapsed_time(pure_idx) = toc;
        epsilon_gd(pure_idx) = epsilon_opt_gd;
        num_iter_gd(i) = num_iter;
        
        fprintf('--- Complete Nystrom Methods, Size %d ---\n', current_size);
    end
    
    fprintf('\n--- All operation complete, Generate Tables... ---\n');

  
    figure('Position', [100, 100, 1200, 600]);

    % Pure Rippa 
    loglog(training_sizes_pure, total_elapsed_time_rippa_pure, '^:', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.9290 0.6940 0.1250], 'DisplayName', 'Rippa');
    hold on;

    % Pure GD 
    loglog(training_sizes_pure, total_elapsed_time_gd_pure, 'd-.', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD');

    % Rippa + Nystrom 
    valid_rippa = ~isnan(total_elapsed_time_rippa);
    loglog(training_sizes_pure(valid_rippa), total_elapsed_time_rippa(valid_rippa), ...
        's--', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Rippa + Nystrom');

    % GD + Nystrom
    valid_gd = ~isnan(total_elapsed_time);
    loglog(training_sizes_pure(valid_gd), total_elapsed_time(valid_gd), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');

    xlabel('Number of Interpolation Nodes (N)', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('Time (seconds)', 'FontSize', 20, 'FontWeight', 'bold');
    title('Time Comparison of Epsilon Optimization Methods (1D)', 'FontSize', 20, 'FontWeight', 'bold');
    legend('show', 'Location', 'northwest', 'FontSize', 18);
    grid on;
    box on;

    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 18);
    xlim([50, 10000]);

   
    figure('Position', [150, 150, 1200, 600]);
    
   
    semilogy(training_sizes_pure, num_iter_gd_pure, 's-', 'Color', [0.5, 0.5, 0.0], ...
        'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Gradient Descent (Pure)');
    hold on;
    
  
    valid_gd_iter = ~isnan(epsilon_gd);
    semilogy(training_sizes_pure(valid_gd_iter), num_iter_gd, 'o-', ...
        'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Gradient Descent + Nystrom');
    
    xlabel('Training Size (N)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Number of Iterations', 'FontSize', 14, 'FontWeight', 'bold');
    title('Number of Iterations for Convergence', 'FontSize', 16, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 12);
    grid on;
    box on;

    set(gca, 'XScale', 'log');
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 12);
    xlim([50, 10000]);

   
    figure('Position', [200, 200, 1200, 600]);
    
   
    plot(training_sizes_pure, epsilon_rippa_pure, 's-', 'Color', [1.0, 0.5, 0.0], ...
        'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Rippa (Pure)');
    hold on;
    
   
    plot(training_sizes_pure, epsilon_gd_pure, 'd-', 'Color', [0.5, 0.5, 0.0], ...
        'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'GD (Pure)');
    
   
    valid_rippa_eps = ~isnan(epsilon_rippa);
    plot(training_sizes_pure(valid_rippa_eps), epsilon_rippa(valid_rippa_eps), ...
        's-', 'Color', 'red', 'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'Rippa + Nystrom');
    
    
    valid_gd_eps = ~isnan(epsilon_gd);
    plot(training_sizes_pure(valid_gd_eps), epsilon_gd(valid_gd_eps), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, 'DisplayName', 'GD + Nystrom');
    
    xlabel('Training Size (N)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Optimal Epsilon', 'FontSize', 14, 'FontWeight', 'bold');
    title('Optimal Shape Parameter Chosen by GD and Rippa', 'FontSize', 16, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 12);
    grid on;
    box on;

    set(gca, 'XScale', 'log');
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 12);
    xlim([50, 10000]);

end
                                                                          

function result = testcase1(x)

    result = 1./(1+16 * (x(:,1).^2));
  %  result = exp(sin(pi.*x(:,1)));
end




function epsilon0 = franke_initial_epsilon(X)
    % Compute diameter of data range in 1D
    D = max(X(:,1)) - min(X(:,1));
    
    % Number of data points
    N = length(X(:,1));
    
    % Franke's formula
    epsilon0 = 0.8 * (N^(1/2)) / D;
end

function phi = gaussian_rbf(r, epsilon) % Generate Gaussian RBF function
  %  phi = exp(-(epsilon * r).^2);
    phi = 1 ./ sqrt(1 + (epsilon * r).^2);
end

function [C, W] = construct_nystrom_matrices(K_full, landmark_indices)
    % Construct Nyström matrices C and W
    % C: n×m matrix (all points to landmarks)
    % W: m×m matrix (landmarks to landmarks)
    
    C = K_full(:, landmark_indices);
    W = K_full(landmark_indices, landmark_indices);
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
function [K_nystrom, W_inv] = compute_nystrom_approximation(C, W, regularization)
    % Compute Nyström approximation
    % K_nystrom = C * W_inv * C'
    
    if nargin < 3
        regularization = 1e-6;
    end
    
    m = size(W, 1);
    W_reg = W + regularization * eye(m);
    
    % Use SVD for stable pseudo-inverse
    [U, S, V] = svd(W_reg);
    s = diag(S);
    s_thresh = max(s, 1e-12);
    W_inv = V * diag(1./s_thresh) * U';
    
    % Compute Nyström approximation
    K_nystrom = C * W_inv * C';
end

function L = compute_loo_error_norm_1d(epsilon, X, f, landmark_indices)
% Woodbury-based LOOCV with Nyström (1D-SPECIFIC VERSION)
% This version is optimized for 1D data by replacing pdist2 with
% broadcasted absolute difference calculations.
%
% A = lambda*I + C*W^{-1}*C', then use Woodbury:
% A^{-1} = lambda^{-1}I - lambda^{-2} C * ( W + lambda^{-1} C'^C )^{-1} * C'
    
    % 0) Tikhonov regularization for numerical stability
    lambda = 1e-10; % Per your provided code

    % 1) Build C, W directly (1D-specific)
    %    X should be N x 1, landmark_indices is m x 1
    Xl = X(landmark_indices, :); % Xl is m x 1

    % --- 1D Specific Distance Calculation ---
    % Use abs(X - Xl') for 1D distances (N x m matrix)
    r_dist_C = abs(X - Xl'); 
    C  = gaussian_rbf(r_dist_C,  epsilon);   % N×m

    % Use abs(Xl - Xl') for 1D distances (m x m matrix)
    r_dist_W = abs(Xl - Xl');
    W  = gaussian_rbf(r_dist_W,  epsilon);   % m×m
    % --- End 1D Specific ---

    % 2) Regularize W slightly (jitter)
    m = size(W,1);
    W = W + 1e-12*eye(m);

    % 3) Form M = W + (1/lambda) * C'^C   (m×m)
    %    Cost is O(N m^2)
    CtC = C.' * C;                 % m×m
    M   = W + (1/lambda) * CtC;    % m×m

    % 4) Factorize M (Cholesky preferred, SVD fallback)
    [R,p] = chol(M);
    if p == 0
        % 4a) Use triangular solve for M^{-1} * (C'^f)
        Cf      = C.' * f;               % m×1
        MinvCf  = R \ (R.' \ Cf);        % m×1

        % 4b) Calculate A^{-1}f
        Ainv_f  = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);   % N×1

        % 4c) Compute Minv for calculating diagonal
        Im = eye(m);
        Minv = R \ (R.' \ Im);           % m×m
    else
        % Fallback to SVD
        [U,S,V] = svd(M);
        s  = diag(S);
        Sinv = diag(1 ./ max(s, 1e-15));
        Minv = V * Sinv * U.';           % m×m

        Cf     = C.' * f;
        MinvCf = Minv * Cf;
        Ainv_f = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);
    end

    % 5) diag(A^{-1}) = lambda^{-1} - lambda^{-2} * rowdot( C*Minv , C )
    T = C * Minv;                                      % N×m
    diagAinv = (1/lambda) - (1/lambda^2) * sum(T .* C, 2);  % N×1
    diagAinv = max(diagAinv, 1e-15);                   % Avoid division by zero

    % 6) Rippa's LOO error vector and objective
    E = Ainv_f ./ diagAinv;                            % N×1
    L = norm(E)^2;
end
function [eta, L_new] = line_search_1d(epsilon, grad, x, f, landmark_indices, eta_ini, gamma, alpha, max_iters, min_eta)
   if nargin < 6, eta_ini = 1.0; end
    if nargin < 7, gamma = 0.01; end  
    if nargin < 8, alpha = 0.5; end   
    if nargin < 9, max_iters = 50; end
    if nargin < 10, min_eta = 1e-12; end 
    
    eta = eta_ini;
    L_current = compute_loo_error_norm_1d(epsilon, x, f,landmark_indices);
    
    for iter = 1:max_iters
        epsilon_new = max(epsilon - eta * grad, 1e-8);  
        L_new = compute_loo_error_norm_1d(epsilon_new, x, f,landmark_indices);
        
       
        if isnan(L_new)
            eta = eta * alpha;
            continue;
        end
        
       
        if L_new <= L_current - gamma * eta * grad^2
            break;
        end
        
        eta = eta * alpha;
        
       
        if eta < min_eta
            eta = min_eta;
            break;
        end
    end
end

function [L, grad] = compute_loo_gradient_1d(epsilon, x, f,landmark_indices)
   
    h = max(1e-8, 1e-8 * abs(epsilon)); 
    L_plus = compute_loo_error_norm_1d(epsilon + h, x, f,landmark_indices);
    L_minus = compute_loo_error_norm_1d(epsilon - h, x, f,landmark_indices);
    grad = (L_plus - L_minus) / (2*h);
    
   
    if isnan(grad) || abs(grad) > 1e10
        grad = sign(grad) * min(abs(grad), 1e10);
    end
    
    L = (L_plus + L_minus)/2; 
end


function [epsilon_opt, history, iter] = optimize_epsilon_gd(X, f, epsilon_init, max_iter, tol, landmark_indices)

    epsilon = max(epsilon_init, 1e-8);
    log_eps = log(epsilon);
    
    history.epsilon = zeros(max_iter + 1, 1);
    history.loss = inf(max_iter + 1, 1);
    

    best_loss = compute_loo_error_norm_1d(epsilon, X,f, landmark_indices);
    
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
        
       
        [L, grad] = compute_loo_gradient_1d(epsilon, X, f, landmark_indices);
        
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
            
            
            L_new = compute_loo_error_norm_1d(epsilon_try, X, f, landmark_indices);
            
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
        E_norms_coarse(idx) = compute_loo_error_norm_1d(grid_coarse(idx), X, f, landmark_indices);
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
        E_norms_fine(idx) = compute_loo_error_norm_1d(grid_fine(idx), X, f, landmark_indices);
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







function [optimal_epsilon, min_E_norm] = optimize_epsilon_pure(X, f)
    
    num_coarse = 30; 
    grid_coarse = logspace(log10(1e-5), log10(1000), num_coarse);
    
    E_norms_coarse = zeros(num_coarse, 1);
    
    for idx = 1:num_coarse
        E_norms_coarse(idx) = compute_loo_error_norm_pure_1d(grid_coarse(idx), X, f);
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
        E_norms_fine(idx) = compute_loo_error_norm_pure_1d(grid_fine(idx), X, f);
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

function L = compute_loo_error_norm_pure_1d(epsilon, x, f)
    N = length(x);
    r_dist = abs(x - x');  % 1D distance matrix
    A = gaussian_rbf(r_dist, epsilon);
    
   
    lambda_reg = 1e-12 * norm(x);
    cond_A = cond(A);
    if cond_A > 1e12
        lambda_reg = lambda_reg * 10;
    end
    A_reg = A + lambda_reg * eye(N);
    
  
    try
        
        [R, p] = chol(A_reg);
        if p == 0
            
            invDiag = sum(inv(R).^2, 2);  % diag(inv(A)) = sum(inv(R).^2, 2)
            lambda_coeff = R \ (R' \ f);
        else
            error('Cholesky failed');
        end
    catch
       
        [U, S, V] = svd(A_reg);
        s = diag(S);
       
        Uf = U' * f;  
      
        s = s(:);  
        lambda_coeff = V * (Uf ./ s); 
        invDiag = sum((V ./ s').^2, 2);  
    end
    
  
    E = lambda_coeff ./ invDiag;
    L = norm(E)^2;
end
function [eta, L_new] = line_search_pure_1d(epsilon, grad, x, f, eta_ini, gamma, alpha, max_iters, min_eta)
    if nargin < 5, eta_ini = 1.0; end
    if nargin < 6, gamma = 0.01; end  
    if nargin < 7, alpha = 0.3; end  
    if nargin < 8, max_iters = 50; end
    if nargin < 9, min_eta = 1e-12; end 
    
    eta = eta_ini;
    L_current = compute_loo_error_norm_pure_1d(epsilon, x, f);
    
    for iter = 1:max_iters
        epsilon_new = max(epsilon - eta * grad, 1e-8);  
        L_new = compute_loo_error_norm_pure_1d(epsilon_new, x, f);
        
       
        if isnan(L_new)
            eta = eta * alpha;
            continue;
        end
        
       
        if L_new <= L_current - gamma * eta * grad^2
            break;
        end
        
        eta = eta * alpha;
        
      
        if eta < min_eta
            eta = min_eta;
            break;
        end
    end
end

function [L, grad] = compute_loo_gradient_pure_1d(epsilon, x, f)
   
    h = max(1e-8, 1e-8 * abs(epsilon));
    L_plus = compute_loo_error_norm_pure_1d(epsilon + h, x, f);
    L_minus = compute_loo_error_norm_pure_1d(epsilon - h, x, f);
    grad = (L_plus - L_minus) / (2*h);
    
  
    if isnan(grad) || abs(grad) > 1e10
        grad = sign(grad) * min(abs(grad), 1e10);
    end
    
    L = (L_plus + L_minus)/2;  
end



function [epsilon_opt, history,iter] = optimize_epsilon_gd_pure(X, f, epsilon_init, max_iter, tol)
   
    epsilon = max(epsilon_init, 1e-8);
    log_eps = log(epsilon);
    
    history.epsilon = zeros(max_iter + 1, 1);
    history.loss = inf(max_iter + 1, 1);
    
  
    best_loss = compute_loo_error_norm_pure_1d(epsilon, X, f);
    
    if isnan(best_loss) || isinf(best_loss)
        fprintf('Error: Initial epsilon for GD-Pure is unstable. Aborting.\n');
        epsilon_opt = epsilon_init;
        history.epsilon(1) = epsilon;
        history.loss(1) = best_loss;
        iter = 0;
        return;
    end
    
    history.loss(1) = best_loss;
    history.epsilon(1) = epsilon;
    
    best_epsilon = epsilon;
    best_log_eps = log_eps;
    
    eta = 1.0; 
    
   
    for iter = 1:max_iter
      
        epsilon = exp(log_eps);
        
     
        [L, grad] = compute_loo_gradient_pure_1d(epsilon, X, f);
        
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
            
           
            L_new = compute_loo_error_norm_pure_1d(epsilon_try, X, f);
            
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