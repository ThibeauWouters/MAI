clear;
close all;
clc; 

addpath('/MATLAB Drive/SVM/FixedSize/');

%% Get data and hyperparameters
X = 3.*randn(100,2);
ssize = 10;
sig2 = 1000;

%% Create subset
subset = zeros(ssize,2);
for t = 1:100

  %
  % new candidate subset
  %
  r = ceil(rand*ssize);
  candidate = [subset([1:r-1 r+1:end],:); X(t,:)];
  
  %
  % is this candidate better than the previous?
  %
  if kentropy(candidate, 'RBF_kernel',sig2)>kentropy(subset, 'RBF_kernel',sig2);
      % YES? Then replace the subset with new candidate
      subset = candidate;
  end
  
  %
  % make a figure
  %
  %plot(X(:,1),X(:,2),'b*'); hold on;
  %plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
  %pause(1)
end

%% Figure at the end:
plot(X(:,1),X(:,2),'b*'); hold on;
plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 