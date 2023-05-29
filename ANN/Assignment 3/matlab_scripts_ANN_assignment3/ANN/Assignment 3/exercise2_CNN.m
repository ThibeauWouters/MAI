clear
clc
close all

%%%%%%%%%%%
% exercise3_CNN
% A script for the discussion of questions regarding CNNs
%%%%%%%%%%%

% Load resnet
convnet = alexnet();
analyzeNetwork(convnet);

% Take a look at the first convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1, 5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure;
montage(w1)

% Inspect layers 1 to 5
% First layer; an input layer, taking 224 x 224 images
layer = convnet.Layers(1);
disp(layer);
inputWidth = layer.InputSize(1);
inputHeight = layer.InputSize(2);
% Second layer: first convolutional layer, 
layer = convnet.Layers(2);
F = layer.FilterSize(1);
P = layer.PaddingSize(1);
S = layer.Stride(1);
outputWidth = ((inputWidth - F + 2*P)/S) + 1;
F = layer.FilterSize(2);
P = layer.PaddingSize(3);
S = layer.Stride(2);
outputHeight = ((inputHeight - F + 2*P)/S) + 1;
disp("The output width  after the first convolutional layer is: " + outputWidth);
disp("The output height after the first convolutional layer is: " + outputHeight);
% Third layer: ReLU layer
layer = convnet.Layers(3);
disp(layer);
% Fourth layer: cross channel normalization layer
layer = convnet.Layers(4);
disp(layer);
layer = convnet.Layers(5);
disp(layer);
F = layer.PoolSize(1);
P = layer.PaddingSize(1);
S = layer.Stride(1);
outputWidth = ((outputWidth - F + 2*P)/S) + 1;
F = layer.PoolSize(2);
P = layer.PaddingSize(3);
S = layer.Stride(2);
outputHeight = ((outputHeight - F + 2*P)/S) + 1;
disp("The output width  after the first pooling layer is: " + outputWidth(1,1));
disp("The output height after the first pooling layer is: " + outputHeight(1,1));

% Inspect the global architecture
for i=1:size(convnet.Layers)
    if contains(convnet.Layers(i).Name, "fc")
        disp("The first fully connected layer is at index: " + i);
        disp("The input size of the fully connected layer is: " + convnet.Layers(i).InputSize);
        break;
    end
end

disp("The original input size, without downsampling, was: " + 227*227*3);
