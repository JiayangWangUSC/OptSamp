clear all;
close all;
clc;

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1/';
dirname = dir(datapath);

%% single image recon
N1 = 320; N2 = 320; Nc = 16; Ns =8;