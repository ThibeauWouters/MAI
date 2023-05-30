close all;
clear;
clc;

%% Experiments on the handwriting data set on kPCA for reconstruction and denoising

load Data/digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

filename = "Data/digitsdn_tuning_results.csv";
header = {"nc", "sig2", "MSE"};
writecell(header, filename);

%% Add noise to the digit maps

noisefactor=1;
noise = noisefactor*maxx; % sd for Gaussian noise

Xn = X; 
for i=1:N
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1; 
for i=1:size(Xtest1,1)
  randn('state', N+i);
  Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

Xnt2 = Xtest2; 
for i=1:size(Xtest2, 1)
  randn('state', N+i);
  Xnt2(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end

%% Denoise using the first principal components

% Training and reconstruction sets
Xtr = X(1:1:end,:);
% CHOOSE TEST SET FOR RECONSTRUCTION HERE
test_set = Xtest2;
Xdt = zeros(size(test_set));

% Choose number of eigenvalues of kpca
npcs = [2.^(0:7) 190];
sig2_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000];

npcs = [2];
sig2_list = 0.001;

for i=1:numel(sig2_list)
    
    for k=1:numel(npcs)
        % Select current parameters
        sig2 = sig2_list(i);
        nc = npcs(k);
        fprintf("Sig2: %0.4f. Nc: %d\n", sig2, nc)
    
        % Do the PCA
        [lam, U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
        [lam, ids]=sort(-lam); 
        lam = -lam; 
        U=U(:,ids);
        Ud=U(:,(1:nc)); 
        lamd=lam(1:nc);
    
        % Select target and denoise it
        Xdt = preimage_rbf(Xtr, sig2, Ud, test_set, 'denoise');
        
        disp("Here!");

        % Compute and save MSE value
        diffs_sq = (Xdt - test_set).^2;
        mse_value = mean(mean(diffs_sq));

        disp("Still here !");
    
        header = {nc, sig2, mse_value};
        writecell(header, filename, 'WriteMode', 'append');
    end
end