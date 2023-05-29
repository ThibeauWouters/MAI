echo off
clear
clc
close all


%%%%%%%%%%%%%%%%%
% exercise2_SAE %
%%%%%%%%%%%%%%%%%

%% Load and preprocess datasets

% Load train and test datasets
load('Files/digittrain_dataset.mat');
load('Files/digittest_dataset.mat');

[~, targets] = max(tTest);

counts = zeros(1, 10);
for i=1:size(counts, 2)
    c = sum(targets == i);
    counts(i)=c;
end
counts/numel(targets)