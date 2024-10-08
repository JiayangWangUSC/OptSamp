%% snr estimation
clear all;
close all;
clc;

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1/';
dirname = dir(datapath);

N1 = 320; N2 = 320; Nc = 16; Ns =8;

%% single image recon
dirnum = 3;
kspace = h5read([datapath,dirname(dirnum).name],'/kspace_central');
kspace = complex(kspace(:,:,1:Nc,1),kspace(:,:,Nc+1:2*Nc,1));
maps = h5read([datapath,dirname(dirnum).name],'/sense_central');
maps = complex(maps(:,:,1:Nc,1),maps(:,:,Nc+1:2*Nc,1));

% signal estimation
im_sense = abs(sum(conj(maps).*ifft2c(kspace),3));
imagesc(im_sense/max(im_sense(:))*2);colormap(gray);clim([0,1]);

%patch = im_sense(170:200,120:150);
%signal_mag = mean(patch(:));

%% def snr
snr = 10;
%sigma = signal_mag/SNR/sqrt(2);
sigma = 0.15/snr; %sigma = 45 SNR=1

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
figure;
imagesc(SNR);
caxis([-0.5,2*snr]);colormap([0,0,0;jet(7)]);colorbar;axis equal;axis tight;

