%% snr estimation
clear all;
close all;
clc;

%% load data
load('sampledata.mat');
[N1,N2,Nc] = size(kspace);

fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

%% sensitivity maps
ncalib = 24; % use 24 calibration lines to compute compression
ksize = [6,6]; % kernel size
eigThresh_1 = 0.1;
eigThresh_2 = 0.95;
calib = crop(kspace,[ncalib,ncalib,Nc]);
[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));
[M,W] = kernelEig(k(:,:,:,1:idx),[N1,N2]);
maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);

%% signal estimation
im_sense = abs(sum(conj(maps).*ifft2c(kspace),3));
%imwrite(im_sense/150,'test.png');
patch = im_sense(170:200,120:150);
%imwrite(patch/150,'test.png');
signal_mag = mean(patch(:));

%% def snr
SNR = 1;
sigma = signal_mag/SNR/sqrt(2);
sigma = 45; %sigma = 45 SNR=1

%%
% Monte Carlo calculation of mean and variance with coil-combination
numTest = 100;
im_series = zeros(N1,N2,numTest);

for i = 1:numTest
    noise = complex(sigma.*randn(N1,N2,Nc),sigma.*randn(N1,N2,Nc));
    im_series(:,:,i) = sum(conj(maps).*ifft2c(kspace+noise),3);
end

me = mean(im_series,3);
Std = std(im_series,0,3);
SNR = abs(me)./Std;
%figure;
%imagesc(SNR);
%caxis([-0.5,2]);colormap([0,0,0;jet(7)]);colorbar;axis equal;axis tight;
