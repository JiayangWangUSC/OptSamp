clear all;
close all;
clc;

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4;

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1_demo/';
dirname = dir(datapath);

%% parameters
N1 = 320; N2 = 320; Nc = 16; Ns =8;
factor = 8;
SNR = 2;
sigma = sqrt(8)*0.12/SNR;  
reso = 0;

beta = 17;  % Higher beta for stronger noise suppression 
N1_wd = N1 - 32*reso;
N2_wd = N2 - 32*reso;
window = kaiser(N1_wd, beta) * kaiser(N2_wd, beta)';

%%
weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * ones(1,N2-32*reso);
%load(['./fftweight_snr',num2str(int8(SNR)),'_reso',num2str(int8(reso))]);

avg_mask = zeros(N1,N2);
avg_mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = repmat(weight,[N1-32*reso,1]);
avg_mask = repmat(avg_mask,[1,1,Nc]);
avg_mask_dagger = avg_mask;
avg_mask_dagger(avg_mask>0) = 1./avg_mask(avg_mask>0);

mask = zeros(N1,N2);
mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = window;
mask = repmat(mask,[1,1,Nc]);

%%
kspace = h5read([datapath,dirname(3).name],'/kspace_central');
Maps = h5read([datapath,dirname(3).name],'/sense_central');
 
kData = complex(kspace(:,:,1:Nc,1),kspace(:,:,Nc+1:2*Nc,1));
maps = complex(Maps(:,:,1:Nc,1),Maps(:,:,Nc+1:2*Nc,1));
gt = abs(sum(ifft2c(kData).*conj(maps),3));

noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
recon = abs(sum(ifft2c(mask.*(kData+sqrt(avg_mask_dagger).*noise)).*conj(maps),3));

%%
% imwrite(recon/max(gt(:))*1.5,['/home/wjy/Project/optsamp_result/opt_fft_snr',num2str(SNR),'.png'])
% imwrite((abs(recon-gt))/max(gt(:))*5,['/home/wjy/Project/optsamp_result/opt_fft_error_snr',num2str(SNR),'.png'])
