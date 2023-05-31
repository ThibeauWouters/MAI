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
%%% CHANGE HERE!
sigmafactor = 0.5;
sig2=sig2*sigmafactor;

make_plot = false;
save_txt = 'Plots/denoise_digits_sig2_large.png';

% Kernel based Principal Component Analysis using the original training data

disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);

% Linear PCA
[lam_lin, U_lin] = pca(Xtr);

% Kernel PCA
% Applied with max 240 principal components
sig2_list = [0.0001, 0.001, 0.5, 100, 200];
sig2_list = [1000000];
for i=1:numel(sig2_list)
    sig2 = sig2_list(i);
    [lam, U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
    [lam, ids]=sort(-lam); 
    lam = -lam; 
    U=U(:,ids);
    
    str = sprintf("Data/digitsdn_lambda_%0.4f.txt", sig2);
    disp(str);
    writematrix(lam, str);
    
end
%writematrix(lam_lin, "Data/digitsdn_lambda_lin.txt");


%% Visualize the ith eigenvector 

 disp(' ');
 disp('Visualize the eigenvectors');
 
 % define the number of eigen vectors to visualize
 nball = min(length(lam),length(lam_lin));
 eigs = [1:10];
 ne=length(eigs); 
 
 % compute the projections of the ith canonical basis vector e_i, i=1:240
 k = kernel_matrix(Xtr,'RBF_kernel',sig2,eye(dim))'; 
 proj_e=k*U;

 
 
 if make_plot
    figure; colormap(gray); eigv_img=zeros(ne,dim);  
 
     for i=1:ne; 
         ieig=eigs(i);
     
         % linear PCA 
         if ieig<=length(lam_lin),
           subplot(3, ne, i); 
           pcolor(1:15,16:-1:1,reshape(real(U_lin(:,ieig)), 15, 16)'); shading interp; 
           set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
           title(['\lambda',sprintf('%d\n%.4f', ieig, lam_lin(ieig))],'fontSize',6) 
           if i==1, ylabel('linear'), end
           drawnow
         end
           
         % kPCA  
         % The preimage of the eigenvector in the feature space might not exist! 
         if ieig<=length(lam),
           eigv_img(i,:) = preimage_rbf(Xtr,sig2,U(:,ieig),zeros(1,dim),'d');
           subplot(3, ne, i+ne); 
           pcolor(1:15,16:-1:1,reshape(real(eigv_img(i,:)),15, 16)'); shading interp;
           set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
           title(['\lambda',sprintf('%d\n%.4f', ieig, lam(ieig))],'fontSize',6) 
           if i==1, ylabel('kernel'); end
           drawnow
         end
     
         if ieig<=size(proj_e,2),
           subplot(3, ne, i+ne+ne); 
           pcolor(1:15,16:-1:1,reshape(real(proj_e(:,ieig)), 15, 16)'); shading interp; 
           set(gca,'xticklabel',[]);set(gca,'yticklabel',[]); 
           title(['\lambda',sprintf('%d\n%.4f', ieig, lam(ieig))],'fontSize',6) 
           if i==1, ylabel('kernel'); end
           drawnow
         end
         
     end
    
    
    %save_txt = 'Plots/denoise_digits_linear.png';
    %exportgraphics(gcf,save_txt, 'Resolution', 600);

 end