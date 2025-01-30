clear all;
close all;
clc;
addpath(genpath("SPIRiT_v0.3"));
%% load data
%datapath = '/home/wjy/Project/fastmri_dataset/brain_T1_demo/';
datapath = '/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_val/';
dirname = dir(datapath);

N1 = 320; N2 = 320; Nc = 16; Ns =8;

%%
ncalib = 24; % use 24 calibration lines to compute compression
ksize = [6,6]; % kernel size

% Threshold for picking singular vercors of the calibration matrix
% (relative to largest singlular value.

eigThresh_1 = 0.02;

% threshold of eigen vector decomposition in image space.
eigThresh_2 = 0.9;

%%
for dnum = 3:length(dirname)

disp([datapath,dirname(dnum).name]);
kspace = h5read([datapath,dirname(dnum).name],'/kspace_central');
Maps = zeros(N1,N2,2*Nc,Ns);   

for ns = 1:Ns
kData = complex(kspace(:,:,1:Nc,ns),kspace(:,:,Nc+1:2*Nc,ns));
 
% crop a calibration area
calib = crop(kData,[ncalib,ncalib,Nc]);

% compute Calibration matrix, perform 1st SVD and convert singular vectors
% into k-space kernels

[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));

[M,W] = kernelEig(k(:,:,:,1:idx),[N1,N2]);
maps = M(:,:,:,end);
maps = cat(3,real(maps),imag(maps));
Maps(:,:,:,ns) = maps;
end

h5write([datapath,dirname(dnum).name],'/sense_central',single(Maps));

end


