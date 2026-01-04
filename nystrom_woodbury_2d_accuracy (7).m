function main()
  
    rng(0);
    
   
    training_sizes_pure = [64, 128, 256, 512, 1024, 2048, 4096, 8192];  
    training_sizes_nystrom = [512, 1024, 2048, 4096, 8192];
    landmark_counts = [200, 200, 200, 200, 200]; 
    
    num_sizes_pure = length(training_sizes_pure);
    num_sizes_nystrom = length(training_sizes_nystrom);


    max_iter = 100;
    tol = 1e-8;
    
    
    reps = 5; 
    maxIterKM = 200; 

   
    elapsed_time_rippa = zeros(num_sizes_nystrom, 1);
    elapsed_time_gd = zeros(num_sizes_nystrom, 1);
    elapsed_time_rippa_pure = zeros(num_sizes_pure, 1);
    elapsed_time_gd_pure = zeros(num_sizes_pure, 1);

   
    num_iter_gd = nan(num_sizes_pure, 1);  
    num_iter_gd_pure = nan(num_sizes_pure, 1); 

  
    mean_error_gd_nystrom = nan(num_sizes_pure, 1);
    mean_error_rippa_nystrom = nan(num_sizes_pure, 1);
    mean_error_rippa_pure = nan(num_sizes_pure, 1);
    mean_error_gd_pure = nan(num_sizes_pure, 1);

    
    X_test = rand(5000, 2);
    f_test = testcase(X_test);
    test_size = length(f_test);

  
    fprintf('\n=== Processing Pure Methods ===\n');
    for i = 1:num_sizes_pure
        N = training_sizes_pure(i);
        
        fprintf('\n--- Pure Methods: Training size N = %d ---\n', N);
        
       
        X_train = rand(N, 2);
        f_train = testcase(X_train);
        
        % Pure Rippa
        tic;
        epsilon_opt_rippa_pure = optimize_epsilon_pure(X_train, f_train);
        elapsed_time_rippa_pure(i) = toc;
        
        % Pure GD
        tic;
        [epsilon_opt_gd_pure, ~, iter_count_gd_pure] = optimize_epsilon_gd_pure(X_train, f_train, franke_initial_epsilon(X_train), max_iter, tol);
        elapsed_time_gd_pure(i) = toc;
        num_iter_gd_pure(i) = iter_count_gd_pure;

       
        A_train_rippa_pure = gaussian_rbf(pdist2(X_train, X_train), epsilon_opt_rippa_pure);
        A_train_rippa_pure = A_train_rippa_pure + 1e-10 * eye(N);
        lambda_train_rippa_pure = A_train_rippa_pure \ f_train;
        
        
        A_train_gd_pure = gaussian_rbf(pdist2(X_train, X_train), epsilon_opt_gd_pure);
        A_train_gd_pure = A_train_gd_pure + 1e-10 * eye(N);
        lambda_train_gd_pure = A_train_gd_pure \ f_train;

       
        A_test_rippa_pure = gaussian_rbf(pdist2(X_test, X_train), epsilon_opt_rippa_pure);
        f_pred_rippa_pure = A_test_rippa_pure * lambda_train_rippa_pure;
        mean_error_rippa_pure(i) = sqrt(norm(f_pred_rippa_pure - f_test)^2 / test_size);
        
       
        A_test_gd_pure = gaussian_rbf(pdist2(X_test, X_train), epsilon_opt_gd_pure);
        f_pred_gd_pure = A_test_gd_pure * lambda_train_gd_pure;
        mean_error_gd_pure(i) = sqrt(norm(f_pred_gd_pure - f_test)^2 / test_size);
        
       
        fprintf('  N=%d, Rippa (Pure)  Mean Error: %.20f\n', N, mean_error_rippa_pure(i));
        fprintf('  N=%d, GD (Pure)     Mean Error: %.20f, Iterations: %d\n', N, mean_error_gd_pure(i), num_iter_gd_pure(i));
    end
    
  
    fprintf('\n=== Processing Nystrom Methods ===\n');
    for i = 1:num_sizes_nystrom
        N = training_sizes_nystrom(i);
        m_landmarks = landmark_counts(i);
        
        
        pure_idx = find(training_sizes_pure == N);
        
        fprintf('\n--- Nystrom Methods: Training size N = %d, landmarks m = %d ---\n', N, m_landmarks);
        
        
        X_train = rand(N, 2);
        f_train = testcase(X_train);
        landmark_indices = kmeans_landmarks_nn(X_train, m_landmarks, reps, maxIterKM);
        
        % Rippa + Nystrom
        tic;
        epsilon_opt_rippa = optimize_epsilon(X_train, f_train, landmark_indices);
        elapsed_time_rippa(i) = toc;
        
        % GD + Nystrom
        tic;
        [epsilon_opt_gd, ~, iter_count_gd] = optimize_epsilon_gd(X_train, f_train, franke_initial_epsilon(X_train), max_iter, tol, landmark_indices);
        elapsed_time_gd(i) = toc;
        num_iter_gd(pure_idx) = iter_count_gd; 

        
        A_train_gd = gaussian_rbf(pdist2(X_train, X_train), epsilon_opt_gd);
        A_train_gd = A_train_gd + 1e-10 * eye(N);
        lambda_train_gd = A_train_gd \ f_train;
        
        
        A_train_rippa = gaussian_rbf(pdist2(X_train, X_train), epsilon_opt_rippa);
        A_train_rippa = A_train_rippa + 1e-10 * eye(N);
        lambda_train_rippa = A_train_rippa \ f_train;

        
        A_test_gd = gaussian_rbf(pdist2(X_test, X_train), epsilon_opt_gd);
        f_pred_gd = A_test_gd * lambda_train_gd;
        mean_error_gd_nystrom(pure_idx) = sqrt(norm(f_pred_gd - f_test)^2 / test_size);
        
       
        A_test_rippa = gaussian_rbf(pdist2(X_test, X_train), epsilon_opt_rippa);
        f_pred_rippa = A_test_rippa * lambda_train_rippa;
        mean_error_rippa_nystrom(pure_idx) = sqrt(norm(f_pred_rippa - f_test)^2 / test_size);
        
       
        fprintf('  N=%d, GD+Nystrom   Mean Error: %.20f, Iterations: %d\n', N, mean_error_gd_nystrom(pure_idx), num_iter_gd(pure_idx));
        fprintf('  N=%d, Rippa+Nystrom Mean Error: %.20f\n', N, mean_error_rippa_nystrom(pure_idx));
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

  
    figure('Position', [100, 100, 1200, 600]);

    % Pure Rippa 
    loglog(training_sizes_pure, mean_error_rippa_pure, '^:', 'LineWidth', 2, ...
        'MarkerSize', 8, 'Color', [0.9290 0.6940 0.1250], 'DisplayName', 'Rippa'); 
    hold on;

    % Pure GD 
    loglog(training_sizes_pure, mean_error_gd_pure, 'd-.', 'LineWidth', 2, ...
        'MarkerSize', 8, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD');

    % Rippa + Nystrom 
    valid_rippa_nystrom = ~isnan(mean_error_rippa_nystrom);
    loglog(training_sizes_pure(valid_rippa_nystrom), mean_error_rippa_nystrom(valid_rippa_nystrom), ...
        's--', 'LineWidth', 2, 'MarkerSize', 8, ...
        'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Rippa + Nystrom');

    % GD + Nystrom 
    valid_gd_nystrom = ~isnan(mean_error_gd_nystrom);
    loglog(training_sizes_pure(valid_gd_nystrom), mean_error_gd_nystrom(valid_gd_nystrom), ...
        'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');

    % Labels / Title
    xlabel('Number of Interpolation Nodes (N)', 'FontSize', 20);
    ylabel('Root Mean Squared Error (RMSE)', 'FontSize', 20);
    title('RMSE Comparison of Epsilon Optimization Methods (2D)', 'FontSize', 20);
    legend('show', 'Location', 'best', 'FontSize', 18);
    grid on;

    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'FontSize', 18);
    
   
    figure('Position', [150, 150, 1200, 600]);
    
    % GD Pure iterations
    semilogx(training_sizes_pure, num_iter_gd_pure, 'd-', 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', [0.4940 0.1840 0.5560], 'DisplayName', 'GD Pure');
    hold on;
    
    % GD Nystrom iterations 
    valid_gd_iter = ~isnan(num_iter_gd);
    semilogx(training_sizes_pure(valid_gd_iter), num_iter_gd(valid_gd_iter), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 10, ...
        'Color', [0 0.4470 0.7410], 'DisplayName', 'GD + Nystrom');
    
    xlabel('Training Size (N)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Number of Iterations', 'FontSize', 14, 'FontWeight', 'bold');
    title('GD Optimization: Iteration Count vs Training Size (2D)', 'FontSize', 16, 'FontWeight', 'bold');
    legend('show', 'Location', 'best', 'FontSize', 12);
    grid on;
    box on;
    
    set(gca, 'XTick', training_sizes_pure);
    set(gca, 'XTickLabel', {'64', '128', '256', '512', '1024', '2048', '4096', '8192'});
    set(gca, 'FontSize', 12);
    xlim([50, 10000]);
    
    fprintf('--- Complete! ---\n');
end

function f = testcase(X)
 x = X(:,1); y = X(:,2);
%  f = (1+exp(-1.0/1.0)-exp(-x/1.0)-exp((x-1.0)/1.0)).*(1+exp(-1.0/1.0)-exp(-y/1.0)-exp((y-1.0)/1.0));
  %  f = (1+exp(-1.0/0.1)-exp(-x/0.1)-exp((x-1.0)/0.1)).*(1+exp(-1.0/0.1)-exp(-y/0.1)-exp((y-1.0)/0.1));
      term1 = 0.75 * exp(-((9*x-2).^2 + (9*y-2).^2)/4);
    term2 = 0.75 * exp(-((9*x+1).^2/49 + (9*y+1)/10));
   term3 = 0.50 * exp(-((9*x-7).^2 + (9*y-3).^2)/4);
   term4 = -0.20 * exp(-(9*x-4).^2 - (9*y-7).^2);
    f = term1 + term2 + term3 + term4;
end

function phi = gaussian_rbf(r, epsilon) 
 
  phi = 1 ./ sqrt(1 + (epsilon * r).^2);
end


function epsilon0 = franke_initial_epsilon(X)
    % Compute diameter of minimal enclosing circle
    [~, radius] = minboundcircle(X(:,1), X(:,2));
    D = 2 * radius;
    
    % Number of data points
    N = size(X, 1);
    
    % Franke's formula
    epsilon0 = 0.8 * (N^(1/4)) / D;
end

function [center, radius] = minboundcircle(x,y)
    % Compute minimal enclosing circle using Welzl's algorithm
    % Input: x,y coordinates of points
    % Output: center and radius of minimal enclosing circle
    
    % Convert to column vectors if needed
    x = x(:);
    y = y(:);
    
    % Remove duplicate points
    xy = unique([x y], 'rows');
    x = xy(:,1);
    y = xy(:,2);
    
    % Start with first point as center
    center = [mean(x), mean(y)];
    radius = max(sqrt((x-center(1)).^2 + (y-center(2)).^2));
    
    % Iteratively improve the circle
    for i = 1:length(x)
        dist = sqrt((x(i)-center(1))^2 + (y(i)-center(2))^2);
        if dist > radius
            % Move center towards this point
            center = center + (dist - radius)/dist * ([x(i), y(i)] - center);
            radius = (radius + dist)/2;
        end
    end
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

function L = compute_loo_error_norm(epsilon, X, landmark_indices, f)
% Woodbury-based LOOCV with Nyström (no explicit N-by-N inverse)
% A = lambda*I + C*W^{-1}*C', then use Woodbury:
% A^{-1} = lambda^{-1}I - lambda^{-2} C * ( W + lambda^{-1} C'^C )^{-1} * C'

   
  
lambda = 1e-6;
    % 1) Build C, W directly (avoid full N×N kernel)
    Xl = X(landmark_indices, :);
    C  = gaussian_rbf(pdist2(X,  Xl),  epsilon);   % N×m
    W  = gaussian_rbf(pdist2(Xl, Xl),  epsilon);   % m×m

  
    m = size(W,1);
    W = W + 1e-12*eye(m);

  
    CtC = C.' * C;                 % m×m
    M   = W + (1/lambda) * CtC;    % m×m

  
    [R,p] = chol(M);
    if p == 0
        
        Cf      = C.' * f;               % m×1
        MinvCf  = R \ (R.' \ Cf);        % m×1

       
        Ainv_f  = (1/lambda) * f - (1/lambda^2) * (C * MinvCf);   % N×1

      
        Im = eye(m);
        Minv = R \ (R.' \ Im);           % m×m
    else
       
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
    diagAinv = max(diagAinv, 1e-15);                  

  
    E = Ainv_f ./ diagAinv;                            % N×1
    L = norm(E)^2;
end



function [L, grad] = compute_loo_gradient(epsilon, X, f,landmark_indices)
    % use numerical gradient
    h = max(1e-8, 1e-8 * abs(epsilon)); % adaptive step size
    L_plus = compute_loo_error_norm(epsilon + h, X, landmark_indices,f);
    L_minus = compute_loo_error_norm(epsilon - h, X,landmark_indices, f);
    grad = (L_plus - L_minus) / (2*h);
    
  
    if isnan(grad) || abs(grad) > 1e10
        grad = sign(grad) * min(abs(grad), 1e10);
    end
    
    L = (L_plus + L_minus)/2; 
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
            break;
        end
        
      
        if abs(log_eps - best_log_eps) < 1e-12 && iter > 5
             % break; 
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


% rippa


function [optimal_epsilon, min_E_norm] = optimize_epsilon(X, f, landmark_indices)
    % optimize_epsilon: Two-Stage Search (Coarse-to-Fine)
  
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






function L = compute_loo_error_norm_pure(epsilon, X, f)
    N = size(X, 1);
    r_dist = pdist2(X, X);
    A = gaussian_rbf(r_dist, epsilon);
    
   
    lambda_reg = 1e-10 * norm(X, 'fro');
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
        invDiag = sum((V ./ (s')).^2, 2);
        lambda_coeff = V * ((U' * f) ./ s);
    end
    
    
    E = lambda_coeff ./ invDiag;
    L = norm(E)^2;
end


function [L, grad] = compute_loo_gradient_pure(epsilon, X, f)
 
    h = max(1e-8, 1e-8 * abs(epsilon)); 
    L_plus = compute_loo_error_norm_pure(epsilon + h, X, f);
    L_minus = compute_loo_error_norm_pure(epsilon - h, X, f);
    grad = (L_plus - L_minus) / (2*h);
    
   
    if isnan(grad) || abs(grad) > 1e10
        grad = sign(grad) * min(abs(grad), 1e10);
    end
    
    L = (L_plus + L_minus)/2; 
end


function [epsilon_opt, history,iter] = optimize_epsilon_gd_pure(X, f, epsilon_init, max_iter, tol)
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
            
            % Armijo 条件
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
    else
        epsilon_opt = exp(log_eps);
    end
    
    history.epsilon = history.epsilon(1:iter+1);
    history.loss = history.loss(1:iter+1);
end