%% data preprocessing for T1 brain data
clear all;
close all;
clc;

%% load data
addpath(genpath('./SPIRiT_v0.3'));
%datapath = '/home/wjy/Project/fastmri_dataset/brain_T1/';
datapath = '/project/jhaldar_118/jiayangw/dataset/brain_T1/multicoil_train/';
dirname = dir(datapath);

%% create new dataset
Nx = 320; Ny = 320; Nc = 20; 

for dir_num = 3:length(dirname)
    h5create([datapath,dirname(dir_num).name],'/kspace_central',[Nx,Ny,2*Nc,8],'Datatype','single');
     h5create([datapath,dirname(dir_num).name],'/sense_central',[Nx,Ny,2*Nc,8],'Datatype','single');
end

%% sense estimation parameters
ncalib = 24; % use 24 calibration lines to compute compression
ksize = [6,6]; % kernel size
eigThresh_1 = 0.1;
eigThresh_2 = 0.95;

%%
for dir_num = 3:length(dirname)
kspace = h5read([datapath,dirname(dir_num).name],'/kspace');
kspace = complex(kspace.r,kspace.i)*2e6;
kspace = kspace(:,:,:,1:8); % select 8 central slices
kspace = permute(kspace,[2,1,3,4]);

kspace_new = zeros(Nx,Ny,Nc,8);
maps_new = zeros(Nx,Ny,Nc,8);

    for s = 1:8
        kdata = kspace(:,:,:,s);
        im = fftshift(ifftn(ifftshift(kdata)));
        im = im(161:480,:,:);
        kdata = fftshift(fft2(ifftshift(im)));
        kspace_new(:,:,:,s) = kdata;
        
        calib = crop(kdata,[ncalib,ncalib,Nc]);
        [k,S] = dat2Kernel(calib,ksize);
        idx = max(find(S >= S(1)*eigThresh_1));
        [M,W] = kernelEig(k(:,:,:,1:idx),[Nx,Ny]);
        maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);
        maps_new(:,:,:,s) = maps;
    end

kData = zeros(Nx,Ny,2*Nc,8);
kData(:,:,1:Nc,:) = real(kspace_new);
kData(:,:,Nc+1:2*Nc,:) = imag(kspace_new);
kData = single(kData);
h5write([datapath,dirname(dir_num).name],'/kspace_central',kData);

Maps = zeros(Nx,Ny,2*Nc,8);
Maps(:,:,1:Nc,:) = real(maps_new);
Maps(:,:,Nc+1:2*Nc,:) = imag(maps_new);
Maps = single(Maps);
h5write([datapath,dirname(dir_num).name],'/sense_central',Maps);


end


