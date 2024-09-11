%% snr estimation
clear all;
close all;
clc;

%% load data
load('sampledata.mat');
[N1,N2,Nc] = size(kspace);

fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4;

%% signal estimation
%% estimate from multicoil image
im_mc = ifft2c(kspace); %multicoil
patch = im_mc(150:200,100:150,:);
signal_mag = mean(abs(patch(:)));

%% estimate from rsos combination
%im = sqrt(sum(abs(ifft2c(kspace)).^2,3)); % rsos combination
%patch = im(150:200,100:150);
%signal_mag = mean(abs(patch(:)));

%% generate noise
SNR = 1;

sigma = signal_mag/SNR/sqrt(2);
noise = complex(sigma.*randn(N1,N2,Nc),sigma.*randn(N1,N2,Nc));

%% add noise
kspace_noisy = kspace + noise;
im_noisy = sqrt(sum(abs(ifft2c(kspace_noisy)).^2,3));
imwrite(im_noisy/max(im_noisy(:))*2,'test.png');

%% validation
coils = conj(im_mc)./sqrt(sum(abs(im_mc).^2,3)); %% sensitivity maps
figure;
imagesc([sqrt(sum(abs(ifft2c(kspace)).^2,3)), abs(sum(coils.*ifft2c(kspace),3))]);
% coil-combined version matches rSoS with "noiseless" data

% Monte Carlo calculation of mean and variance with coil-combination
numTest = 1000;
im_series = zeros(N1,N2,numTest);
for i = 1:numTest
    noise = complex(sigma.*randn(N1,N2,Nc),sigma.*randn(N1,N2,Nc));
    im_series(:,:,i) = sum(coils.*ifft2c(kspace+noise),3);
end
me = mean(im_series,3);
std = std(im_series,0,3);
SNR = abs(me)./std;
figure;
imagesc(SNR);
caxis([-0.5,7.5]);colormap([0,0,0;jet(7)]);colorbar;axis equal;axis tight;
% SNR is about 5
