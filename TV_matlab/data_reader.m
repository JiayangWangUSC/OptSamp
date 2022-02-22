%%
clear all;
close all;
clc;

datapath = '/home/wjy/Project/fastmri_dataset/test/';


dirname = dir(datapath);


%data = h5read('file_brain_AXT2_200_6002217.h5','/home/wjy/Project/fastmri_dataset/test');
kspace = h5read([datapath,dirname(3).name],'/kspace');
kspace = complex(kspace.r,kspace.i);
kspace = permute(kspace,[4,3,2,1]);

fft2c = @(x) fftshift(fft2(ifftshift(x)));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)));

subject = [];
slice = [];

for i = 3:length(dirname)
    fname = dirname(i).name;
    kspace = h5read([datapath,fname],'/kspace');
    kspace = complex(kspace.r,kspace.i);
    kspace = permute(kspace,[4,3,2,1]);
    for snum = 1:size(kspace,1)
        subject = [subject;fname];
        slice = [slice;snum];
    end
end

datalen = length(slice);
batch_size = 8;
batch_num = datalen/batch_size;


for 
