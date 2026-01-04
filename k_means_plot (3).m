%% K-means NMI stability test - Multiple Dimensions and Landmark Sizes

% --- Parameters ---
N_values = [1024, 2048, 4096, 8192]; 
K_values = [50, 100, 200, 300, 400];   
R        = 10;                        
D_values = [1, 2, 3];                 

% --- Storage for results ---
% 结构：nmi_results{d_idx, k_idx} = [means, stds] for each N
nmi_results = cell(length(D_values), length(K_values));


rng(1);

% --- Main loop over D and K ---
for d_idx = 1:length(D_values)
    D = D_values(d_idx);
    
    for k_idx = 1:length(K_values)
        K = K_values(k_idx);
        
        fprintf('===================================\n');
        fprintf('Running for D=%d, K=%d\n', D, K);
        
        nmi_means = zeros(length(N_values), 1);
        nmi_stds  = zeros(length(N_values), 1);
        
        % Loop over different dataset sizes N
        for n_idx = 1:length(N_values)
            N = N_values(n_idx);
            fprintf('  N=%d...\n', N);
            
            % 1. Generate data (uniform in [0,1]^D)
            rng(1);  
            X = rand(N, D);
            
            % 2. Multiple K-means runs with different seeds
            all_idx = cell(R, 1);
            opts = statset('MaxIter', 1000);
            
            for r = 1:R
                rng(r);  
                all_idx{r} = kmeans(X, K, ...
                                    'Start',    'plus', ...
                                    'Options',  opts, ...
                                    'Replicates', 1);
            end
            
            % 3. Compute pairwise NMI
            num_pairs = R * (R - 1) / 2;
            NMIs      = zeros(num_pairs, 1);
            ptr       = 1;
            
            for i = 1:R
                for j = i+1:R
                    idx1 = all_idx{i};
                    idx2 = all_idx{j};
                    NMIs(ptr) = normalizedMutualInformation(idx1, idx2);
                    ptr = ptr + 1;
                end
            end
            
            % 4. Store mean and std
            nmi_means(n_idx) = mean(NMIs);
            nmi_stds(n_idx)  = std(NMIs);
        end
        
        % Store results for this D and K combination
        nmi_results{d_idx, k_idx} = [nmi_means, nmi_stds];
        
        fprintf('D=%d, K=%d completed.\n', D, K);
    end
end

fprintf('===================================\n');
fprintf('All runs completed. Plotting results...\n');

% --- 5. Plot: Separate figure for each K value ---
colors = [0.0, 0.45, 0.74;    % Blue for 1D
          0.85, 0.33, 0.10;   % Red for 2D
          0.93, 0.69, 0.13];  % Yellow/Gold for 3D

for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    
    
    figure('Position', [100 + (k_idx-1)*50, 100 + (k_idx-1)*50, 600, 500]);
    hold on;
    
    % Plot each dimension
    for d_idx = 1:length(D_values)
        D = D_values(d_idx);
        results = nmi_results{d_idx, k_idx};
        nmi_means = results(:, 1);
        nmi_stds  = results(:, 2);
        
        errorbar(N_values, nmi_means, nmi_stds, '-o', ...
            'LineWidth', 2, ...
            'MarkerSize', 8, ...
            'CapSize', 8, ...
            'MarkerFaceColor', colors(d_idx, :), ...
            'Color', colors(d_idx, :), ...
            'DisplayName', sprintf('%dD', D));
    end
    
    set(gca, 'XScale', 'log');
    set(gca, 'XTick', N_values);
    set(gca, 'XTickLabel', arrayfun(@num2str, N_values, 'UniformOutput', false));
    xlim([min(N_values)*0.9, max(N_values)*1.1]);
    ylim([0.5, 1.0]); 
    
    xlabel('Dataset Size N', 'FontSize', 12);
    ylabel('Mean NMI (\pm Std Dev)', 'FontSize', 12);
    title(sprintf('K-means Stability with K=%d Landmarks (R=%d runs)', K, R), ...
          'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    box on;
    legend('Location', 'southwest', 'FontSize', 11);
    hold off;
    
   
    % saveas(gcf, sprintf('kmeans_stability_K%d.png', K));
end

fprintf('All figures created successfully!\n');


%% ============== Helper functions ==============


function nmi = normalizedMutualInformation(l1, l2)
    l1 = l1(:); 
    l2 = l2(:);
    n  = numel(l1);

   
    G  = accumarray([l1 l2], 1);
    pi = sum(G, 2) / n;      
    pj = sum(G, 1) / n;      
    P  = G / n;             
   
    nz    = P > 0;
    denom = pi * pj;       
    MI    = sum( P(nz) .* log( P(nz) ./ denom(nz) ) );

   
    H1 = -sum( pi(pi>0) .* log( pi(pi>0) ) );
    H2 = -sum( pj(pj>0) .* log( pj(pj>0) ) );

   
    if H1 + H2 == 0
        nmi = 1;
    else
        nmi = 2 * MI / (H1 + H2);
    end
end