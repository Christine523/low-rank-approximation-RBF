function main()

   
    rng(0);
    
    
    training_sizes_pure = [64, 128, 256, 512, 1024, 2048, 4096, 8192];  
    training_sizes_nystrom = [512, 1024, 2048, 4096, 8192]; 
    landmark_counts = [200, 200, 200, 200, 200];  
    
    num_sizes_pure = length(training_sizes_pure);
    num_sizes_nystrom = length(training_sizes_nystrom);
    
    reps = 5; 
    maxIterKM = 200;
    tol = 1e-8;
    gd_max_iter = 1000;

   
    epsilon_opt_gd = nan(1, num_sizes_pure);
    epsilon_opt_rippa = nan(1, num_sizes_pure);
    epsilon_opt_rippa_pure = nan(1, num_sizes_pure);
    epsilon_opt_gd_pure = nan(1, num_sizes_pure);
    
 
    num_iter_gd = nan(1, num_sizes_pure);
    num_iter_gd_pure = nan(1, num_sizes_pure);
    
   
    x_train_sets = cell(1, num_sizes_pure);
    f_train_sets = cell(1, num_sizes_pure);
    
   
    lambda_train_gd = cell(1, num_sizes_pure);
    lambda_train_rippa = cell(1, num_sizes_pure);
    lambda_train_rippa_pure = cell(1, num_sizes_pure);
    lambda_train_gd_pure = cell(1, num_sizes_pure);

    
  
    fprintf('\n=== Processing Pure Methods ===\n');
    
    for i = 1:num_sizes_pure
        current_size = training_sizes_pure(i);
        
        fprintf('\n--- Pure Methods: Training size N = %d (%d/%d) ---\n', current_size, i, num_sizes_pure);

   
        fprintf('  [1/3] Generate Data...\n');
        x_train = rand(current_size, 1);
        f_train = testcase1(x_train);
        
       
        x_train_sets{i} = x_train;
        f_train_sets{i} = f_train;

     
        fprintf('  [2/3] Optimize Epsilon (Pure methods)...\n');
        
        % Rippa Pure
        epsilon_opt_rippa_pure(i) = optimize_epsilon_pure(x_train, f_train);
        
        % GD Pure 
        [epsilon_opt_gd_pure(i), ~, num_iter_gd_pure(i)] = optimize_epsilon_gd_pure(x_train, f_train, franke_initial_epsilon(x_train), gd_max_iter, tol);

    
        fprintf('  [3/3] Compute Lambda Coefficient (Pure methods)...\n');
        
        % Rippa Pure
        A_train_rippa_pure = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_rippa_pure(i));
        A_train_rippa_pure = A_train_rippa_pure + 1e-6 * eye(size(A_train_rippa_pure));
        lambda_train_rippa_pure{i} = A_train_rippa_pure \ f_train;
        
        % GD Pure
        A_train_gd_pure = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_gd_pure(i));
        A_train_gd_pure = A_train_gd_pure + 1e-6 * eye(size(A_train_gd_pure));
        fprintf('    Cond(A_gd_pure) for size %d: %e\n', current_size, cond(A_train_gd_pure));
        lambda_train_gd_pure{i} = A_train_gd_pure \ f_train;
        
        fprintf('    GD Pure iterations: %d\n', num_iter_gd_pure(i));
        fprintf('--- Complete Pure Methods, Size %d ---\n', current_size);
    end
    
  
    fprintf('\n=== Processing Nystrom Methods ===\n');
    
    for i = 1:num_sizes_nystrom
        current_size = training_sizes_nystrom(i);
        current_m = landmark_counts(i);
        
       
        pure_idx = find(training_sizes_pure == current_size);
        
        fprintf('\n--- Nystrom Methods: Training size N = %d, landmarks m = %d (%d/%d) ---\n', ...
            current_size, current_m, i, num_sizes_nystrom);

      
        fprintf('  [1/3] Generate data and landmarks...\n');
        x_train = rand(current_size, 1);
        f_train = testcase1(x_train);
        landmark_indices = kmeans_landmarks_nn(x_train, current_m, reps, maxIterKM);
        
        
        x_train_sets{pure_idx} = x_train;
        f_train_sets{pure_idx} = f_train;

       
        fprintf('  [2/3] Optimize Epsilon (Nystrom methods)...\n');
        
        % Rippa with Nystrom
        epsilon_opt_rippa(pure_idx) = optimize_epsilon(x_train, f_train, landmark_indices);
        
        % GD with Nystrom 
        [epsilon_opt_gd(pure_idx), ~, num_iter_gd(pure_idx)] = optimize_epsilon_gd(x_train, f_train, franke_initial_epsilon(x_train), gd_max_iter, tol, landmark_indices);

       
        fprintf('  [3/3] Compute Lambda coefficient (Nystrom methods)...\n');
        
        % GD Nystrom
        A_train_gd = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_gd(pure_idx));
        A_train_gd = A_train_gd + 1e-14 * eye(size(A_train_gd));
        fprintf('    Cond(A_gd_nystrom) for size %d: %e\n', current_size, cond(A_train_gd));
        lambda_train_gd{pure_idx} = A_train_gd \ f_train;
        
        % Rippa Nystrom
        A_train_rippa = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_rippa(pure_idx));
        A_train_rippa = A_train_rippa + 1e-14 * eye(size(A_train_rippa));
        lambda_train_rippa{pure_idx} = A_train_rippa \ f_train;
        
       
        A_train_rippa_pure = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_rippa_pure(pure_idx));
        A_train_rippa_pure = A_train_rippa_pure + 1e-14 * eye(size(A_train_rippa_pure));
        lambda_train_rippa_pure{pure_idx} = A_train_rippa_pure \ f_train;
        
        A_train_gd_pure = gaussian_rbf(pdist2(x_train, x_train), epsilon_opt_gd_pure(pure_idx));
        A_train_gd_pure = A_train_gd_pure + 1e-14 * eye(size(A_train_gd_pure));
        lambda_train_gd_pure{pure_idx} = A_train_gd_pure \ f_train;
        
        fprintf('    GD Nystrom iterations: %d\n', num_iter_gd(pure_idx));
        fprintf('--- Complete Nystrom Methods, Size %d ---\n', current_size);
    end
    
    fprintf('\n--- Training Complete. Start Evaluating... ---\n');

    
  
    X_test = rand(5000, 1);
    f_test = testcase1(X_test);
    test_size = length(f_test);

 
    mean_training = nan(1, num_sizes_pure);
    mean_training_rippa = nan(1, num_sizes_pure);
    mean_training_rippa_pure = nan(1, num_sizes_pure);
    mean_training_gd_pure = nan(1, num_sizes_pure);
    
 
    for i = 1:num_sizes_pure
        current_size = training_sizes_pure(i);
        
        
        if isempty(x_train_sets{i})
            continue;
        end
        
        fprintf('  Start Evaluating Training Size: %d\n', current_size);

        
        x_train = x_train_sets{i};
        
       
        if ~isnan(epsilon_opt_gd(i))
            lambda_gd = lambda_train_gd{i};
            eps_gd = epsilon_opt_gd(i);
            A_test_gd = gaussian_rbf(pdist2(X_test, x_train), eps_gd);
            f_pred_gd = A_test_gd * lambda_gd;
            mean_training(i) = sqrt(norm(f_pred_gd - f_test)^2 / test_size);
        end
        
        
        if ~isnan(epsilon_opt_rippa(i))
            lambda_rippa = lambda_train_rippa{i};
            eps_rippa = epsilon_opt_rippa(i);
            A_test_rippa = gaussian_rbf(pdist2(X_test, x_train), eps_rippa);
            f_pred_rippa = A_test_rippa * lambda_rippa;
            mean_training_rippa(i) = sqrt(norm(f_pred_rippa - f_test)^2 / test_size);
        end
        
      
        lambda_rippa_pure = lambda_train_rippa_pure{i};
        eps_rippa_pure = epsilon_opt_rippa_pure(i);
        A_test_rippa_pure = gaussian_rbf(pdist2(X_test, x_train), eps_rippa_pure);
        f_pred_rippa_pure = A_test_rippa_pure * lambda_rippa_pure;
        mean_training_rippa_pure(i) = sqrt(norm(f_pred_rippa_pure - f_test)^2 / test_size);
        
       
        lambda_gd_pure = lambda_train_gd_pure{i};
        eps_gd_pure = epsilon_opt_gd_pure(i);
        A_test_gd_pure = gaussian_rbf(pdist2(X_test, x_train), eps_gd_pure);
        f_pred_gd_pure = A_test_gd_pure * lambda_gd_pure;
        mean_training_gd_pure(i) = sqrt(norm(f_pred_gd_pure - f_test)^2 / test_size);
        
       
        if ~isnan(mean_training(i))
            fprintf('  mean error norm (GD Nystrom): %.20f\n', mean_training(i));
        end
        if ~isnan(mean_training_rippa(i))
            fprintf('  mean error norm (Rippa Nystrom): %.20f\n', mean_training_rippa(i));
        end
        fprintf('  mean error norm (Rippa Pure): %.20f\n', mean_training_rippa_pure(i));
        fprintf('  mean error norm (GD Pure): %.20f\n', mean_training_gd_pure(i));
    end

  
    fprintf('\n=== GD Iteration Summary ===\n');
    fprintf('Training Size | GD Pure Iters | GD Nystrom Iters\n');
    fprintf('---------------------------------------------------\n');
    for i = 1:num_sizes_pure
        if ~isnan(num_iter_gd(i))
            fprintf('     %5d    |      %4d      |       %4d\n', ...
                training_sizes_pure(i), num_iter_gd_pure(i), num_iter_gd(i));
        else
            fprintf('     %5d    |      %4d      |       N/A\n', ...
                training_sizes_pure(i), num_iter_gd_pure(i));
        end
    end

    % --- 绘图 (阶段 4) ---
    fprintf('\n--- Evaluation complete。Generating figures... ---\n');
    
   
    figure('Position', [100, 100, 1200, 600]);
    
   
    loglog(training_sizes_pure, mean_training_rippa_pure, '^:', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.9290 0.6940 0.1250], 'DisplayName', 'Rippa');
    hold on;
    
    
    loglog(training_sizes_pure, mean_training_gd_pure, 'd-.', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD');
    
 
    valid_rippa = ~isnan(mean_training_rippa);
    loglog(training_sizes_pure(valid_rippa), mean_training_rippa(valid_rippa), ...
        's--', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Rippa + Nystrom');
    
   
    valid_gd = ~isnan(mean_training);
    loglog(training_sizes_pure(valid_gd), mean_training(valid_gd), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');
    
    xlabel('Number of Interpolation Nodes (N)', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('Root Mean Squared Error (RMSE)', 'FontSize', 20, 'FontWeight', 'bold');
    title('RMSE Comparison of Epsilon Optimization Methods (1D)', 'FontSize', 20, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 18);
    grid on;
    box on;
    
    set(gca, 'XScale', 'log');
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 18);
    xlim([50, 10000]);
    
   
    figure('Position', [150, 150, 1200, 600]);
    
   
    semilogx(training_sizes_pure, num_iter_gd_pure, 'd-', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD Pure');
    hold on;
    
   
    valid_gd_iter = ~isnan(num_iter_gd);
    semilogx(training_sizes_pure(valid_gd_iter), num_iter_gd(valid_gd_iter), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');
    
    xlabel('Training Size (N)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Number of Iterations', 'FontSize', 14, 'FontWeight', 'bold');
    title('GD Optimization: Iteration Count vs Training Size', 'FontSize', 16, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 12);
    grid on;
    box on;
    
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 12);
    xlim([50, 10000]);
    
    fprintf('--- Complete! ---\n');

end


% ==================== 辅助函数 ====================

function result = testcase1(x)
    
    result = exp(sin(pi.*x(:,1)));
 %   result = 1./(1+16 .* x(:,1).^2);
end

function epsilon0 = franke_initial_epsilon(X)
    % Compute diameter of data range in 1D
    D = max(X(:,1)) - min(X(:,1));
    
    % Number of data points
    N = length(X(:,1));
    
    % Franke's formula
    epsilon0 = 0.8 * (N^(1/2)) / D;
end

function phi = gaussian_rbf(r, epsilon)
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

function L = compute_loo_error_norm_1d(epsilon, X, f, landmark_indices)
    lambda = 1e-10;
    Xl = X(landmark_indices, :);
    r_dist_C = abs(X - Xl'); 
    C  = gaussian_rbf(r_dist_C,  epsilon);
    r_dist_W = abs(Xl - Xl');
    W  = gaussian_rbf(r_dist_W,  epsilon);
    m = size(W,1);
    W = W + 1e-12*eye(m);
    CtC = C.' * C;
    M   = W + (1/lambda) * CtC;
    [R,p] = chol(M);
    if p == 0
        Cf      = C.' * f;
        MinvCf  = R \ (R.' \ Cf);
        Ainv_f  = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);
        Im = eye(m);
        Minv = R \ (R.' \ Im);
    else
        [U,S,V] = svd(M);
        s  = diag(S);
        Sinv = diag(1 ./ max(s, 1e-15));
        Minv = V * Sinv * U.';
        Cf     = C.' * f;
        MinvCf = Minv * Cf;
        Ainv_f = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);
    end
    T = C * Minv;
    diagAinv = (1/lambda) - (1/lambda^2) * sum(T .* C, 2);
    diagAinv = max(diagAinv, 1e-15);
    E = Ainv_f ./ diagAinv;
    L = norm(E)^2;
end

function [L, grad] = compute_loo_gradient_1d(epsilon, x, f, landmark_indices)
    h = max(1e-8, 1e-8 * abs(epsilon));
    L_plus = compute_loo_error_norm_1d(epsilon + h, x, f, landmark_indices);
    L_minus = compute_loo_error_norm_1d(epsilon - h, x, f, landmark_indices);
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
    
    best_loss = compute_loo_error_norm_1d(epsilon, X, f, landmark_indices);
    
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
        
        grad_log = grad * epsilon;
        
        if abs(grad_log) < tol
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
            break;
        end
    end
    
    if iter > 0 && history.loss(iter+1) > best_loss
        epsilon_opt = best_epsilon;
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
    
    for idx = 1:num_coarse
        E_norms_coarse(idx) = compute_loo_error_norm_1d(grid_coarse(idx), X, f, landmark_indices);
    end
    
    [min_val_coarse, min_idx_coarse] = min(E_norms_coarse);
    
    if isinf(min_val_coarse) || isnan(min_val_coarse)
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
end

function L = compute_loo_error_norm_pure_1d(epsilon, x, f)
    N = length(x);
    r_dist = abs(x - x');
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
            invDiag = sum(inv(R).^2, 2);
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

function [epsilon_opt, history, iter] = optimize_epsilon_gd_pure(X, f, epsilon_init, max_iter, tol)
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
        
        grad_log = grad * epsilon;
        
        if abs(grad_log) < tol
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
            break;
        end
    end
    
    if iter > 0 && history.loss(iter+1) > best_loss
        epsilon_opt = best_epsilon;
    else
        epsilon_opt = exp(log_eps);
    end
    
    history.epsilon = history.epsilon(1:iter+1);
    history.loss = history.loss(1:iter+1);
end