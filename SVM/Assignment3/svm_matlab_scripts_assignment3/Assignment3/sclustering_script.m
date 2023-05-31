clear;
close all;

load two3drings;        % load the toy example

[N,d]=size(X);

perm=randperm(N);   % shuffle the data
X=X(perm,:);

%%% Pick hyperparam here! %%%
sig2=1;              % set the kernel parameters
%%% ===================== %%%


K=kernel_matrix(X,'RBF_kernel',sig2);   %compute the RBF kernel (affinity) matrix

D=diag(sum(K));         % compute the degree matrix (sum of the columns of K)

[U,lambda]=eigs(inv(D)*K,3);  % Compute the 3 largest eigenvalues/vectors using Lanczos
                              % The largest eigenvector does not contain
                              % clustering information. For binary clustering,
                              % the solution is the second largest eigenvector.
                              
clust=sign(U(:,2)); % Threshold the eigenvector solution to obtain binary cluster indicators

[y,order]=sort(clust,'descend');    % Sort the data using the cluster information
Xsorted=X(order,:);

Ksorted=kernel_matrix(Xsorted,'RBF_kernel',sig2);   % Compute the kernel matrix of the
                                                    % sorted data.


proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second,
                    % and third largest eigenvectors.
                                                    
                                                   
%%%% PLOTTING SECTION %%%% 


subplot(1,2,1)
scatter3(X(:,1),X(:,2),X(:,3),15);
title('Data');

subplot(1,2,2);
scatter3(X(:,1),X(:,2),X(:,3),30,clust);
title('Clustering results');

title_text = sprintf("Plots/sclusterting_data_%0.4f.png", sig2);
exportgraphics(gcf,title_text, 'Resolution', 600);

%disp('<<<<<<<<<<<<Press any key>>>>>>>>>>>>>>');
%pause;


figure;
subplot(1,2,1);
imshow(K);
title('Original kernel matrix');

subplot(1,2,2);
imshow(Ksorted);
title('Kernel matrix after sorting');
title_text = sprintf("Plots/sclusterting_kernel_matrix_%0.4f.png", sig2);
exportgraphics(gcf,title_text, 'Resolution', 600);

figure;
scatter(proj(:,1),proj(:,2),15,clust);
title('Projections onto subspace');
title_text = sprintf("Plots/sclusterting_projections_%0.4f.png", sig2);
exportgraphics(gcf,title_text, 'Resolution', 600);