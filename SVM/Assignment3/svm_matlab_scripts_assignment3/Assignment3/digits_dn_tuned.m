close all;
clear;
clc;

%% Experiments on the handwriting data set on kPCA for reconstruction and denoising

load Data/digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

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

%% Select training set
Xtr = X(1:1:end,:);

% Rule of thumb
sig2 =dim*mean(var(Xtr)); 
sigmafactor = 0.7;
sig2=sig2*sigmafactor;
save_txt = 'Plots/denoise_digits_sig2_large.png';

% Kernel based Principal Component Analysis using the original training data

disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);

% Linear PCA
[lam_lin,U_lin] = pca(Xtr);

% Kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

%% Denoise using the first principal components

% choose the digits for test
digs=[0:9]; ndig=length(digs);
m=2; % Choose the mth data for each digit 

Xdt=zeros(ndig,dim);

%% Denoise

% Choose number of eigenvalues of kpca
npcs = [2.^(0:7) 190];
lpcs = length(npcs);
sig2_list = [0.001, 0.01, 0.1, 1, 10, 100];

filename = 'Data/digitsdn_tuned.csv';
header = {'n_components','Sig2','TrainMSE','MSE1', 'MSE2'};
writecell(header, filename);

for i=1:length(npcs)
    for j=1:length(sig2_list)
        % Get hyperparams
        sig2 = sig2_list(j);
        nb_pcs = npcs(i);
        % Fit and denoise
        Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
        Xtr_rec = preimage_rbf(Xtr,sig2,Ud,Xtr,'denoise');
        Xdt_1 = preimage_rbf(Xtr,sig2,Ud,Xtest1,'denoise');
        Xdt_2 = preimage_rbf(Xtr,sig2,Ud,Xtest2,'denoise');
        % Get errors
        mse_tr = mean(mean((Xtr- Xtr_rec).^2));
        mse_1 = mean(mean((Xtest1 - Xdt_1).^2));
        mse_2 = mean(mean((Xtest2 - Xdt_2).^2));
        % Write
        header = {nb_pcs,sig2,mse_tr,mse_1,mse_2};
        writecell(header, filename, "WriteMode", "append");
    end
end





