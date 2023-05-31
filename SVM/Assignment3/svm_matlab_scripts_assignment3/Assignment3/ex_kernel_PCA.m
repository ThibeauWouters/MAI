clc;
clear;
close all;

%% Set up hyperparameters
% DON'T CHANGE THESE
nb = 400;
sig = 0.3;

nb=nb/2;

filename = 'Data/yin_yang_tuning.csv';
header = {'nc','sig2','MSE'};
writecell(header, filename);

% Construct data
leng = 1;
for t=1:nb
  yin(t,:)         = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:)        = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end

% Get as dataset
X = [samplesyin;samplesyang];

% Cross validation (train: 80%, validation: 20%)
cv = cvpartition(size(X,1),'HoldOut', 0.2);
idx = cv.test;
% Separate to training and test data
Xtrain = X(~idx,:);
Xtest = X(idx,:);

% plot dataset
%h=figure; hold on
%plot(samplesyin(:,1),samplesyin(:,2),'o');
%plot(samplesyang(:,1),samplesyang(:,2),'o');
%xlabel('X_1');
%ylabel('X_2');
%title('Yin Yang dataset');

%% Denoise the data by minimizing the reconstruction error

sig2_list = [0.001, 0.01, 0.1, 1, 10, 100];
nc_list = [3, 5, 10, 15];

sig2_list = [0.4];
nc_list = [50];

show_plot = false;

for i=1:length(sig2_list)
    sig2 = sig2_list(i);
    for j=1:length(nc_list)
        nc = nc_list(j);

        fprintf("Sig2: %0.4f, nc: %d\n", sig2, nc);

        % Train PCA:
        [lam, U] = kpca(Xtrain,'RBF_kernel',sig2,[],'eigs',nc);


        %%% NOT WORKING
        % Xd = denoise_kpca(Xtrain,'RBF_kernel',sig2,Xtest,'eigs',nc);
        
        %Xd = preimage_rbf(X,sig2,U(:,1:nc), Xtest,'d', nc);
        

        %%% Full dataset:
        %Xd = denoise_kpca(X,'RBF_kernel',sig2,[],approx,nc);
        %mse = mean(mean((X - Xd).^2));

        % Denoise validation set
        % Code taken from digitsdn
        [lam, ids]=sort(-lam); 
        lam = -lam; 
        U=U(:,ids);
        Ud=U(:,(1:nc)); 
        lamd=lam(1:nc);
        Xd = preimage_rbf(Xtrain, sig2, Ud, Xtest, 'denoise');
        mse_value = mean(mean((Xtest - Xd).^2));

        % Save:
        header = {nc,sig2,mse_value};
        writecell(header, filename, 'WriteMode', 'append');
    end
end

%% Create a plot
title_str = sprintf("%d components", nc);
if show_plot
    figure;
    hold on;
    plot(samplesyin(:,1),samplesyin(:,2),'bo');
    plot(samplesyang(:,1),samplesyang(:,2),'bo');
    plot(Xd(:,1), Xd(:,2),'r+');
    hold off;
    title(title_str);
    str = sprintf('Plots/yin_yang_%d.png', nc);
    disp("Saving to");
    disp(str);
    exportgraphics(gcf,str, 'Resolution', 600);
end
