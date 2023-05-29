%% Load and Explore the Image Data
% Load the digit sample data as an |ImageDatastore| object.

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

%%
% Check the number of images in each category. 
CountLabel = digitData.countEachLabel;


%% 
% Specify the size of the images in the input layer of the
% network. 
% Check the size of the first image in |digitData| .
img = readimage(digitData,1);
size(img)

%% Specify Training and Test Sets
trainingNumFiles = 750;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
				trainingNumFiles,'randomize'); 


%% Define the Network Layers
% Define the convolutional neural network architecture. 
% layers = [imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer()];  %+-3min
      
layers = [imageInputLayer([28 28 1])
  convolution2dLayer(5,24)
  reluLayer
  
  maxPooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(5,48)
  reluLayer  
  
  %fullyConnectedLayer(100)

  fullyConnectedLayer(10)
  softmaxLayer
  classificationLayer()]; %+-10min
     
%% Training options
options = trainingOptions('sgdm','MaxEpochs',15, ...
	'InitialLearnRate',0.0001);  

%% Train the Network Using Training Data
% Train the network you defined in layers, using the training data and the
% training options you defined in the previous steps.
tic
convnet = trainNetwork(trainDigitData,layers,options);
toc


%% 
% Calculate the accuracy. 
accuracy = sum(YTest == TTest)/numel(TTest);

%%
% Accuracy is the ratio of the number of true labels in the test data
% matching the classifications from classify, to the number of images in
% the test data. In this case about 98% of the digit estimations match the
% true digit values in the test set.
