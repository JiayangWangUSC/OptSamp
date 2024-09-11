clear all;
close all;
clc;

%% load data

fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

datapath = '/home/wjy/Project/fastmri_dataset/miniset_brain/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16;

subject = [];
slice = [];
norm_coef = [];

for i = 3:length(dirname)
    fname = dirname(i).name;
    kspace = h5read([datapath,fname],'/kspace');
    kspace = complex(kspace.r,kspace.i);
    im = sqrt(sum(abs(ifft2c(kspace(:,:,:,1))).^2,3));
    kspace = permute(kspace,[4,2,1,3]);
    central_norm = 0.5*max(im(:));
    for snum = 1:size(kspace,1)
        subject = [subject;fname];
        norm_coef = [norm_coef; central_norm];
        slice = [slice;snum];
    end
end

%% difference 
d1 = diag(ones(N1,1));
for n = 1:N1-1
    d1(n,n+1) = -1;    
end
d1(N1,1) = -1;

d2 = diag(ones(N2,1));
for n = 1:N2-1
    d2(n,n+1) = -1; 
end
d2(N2,1) = -1;


D = @(x) difference(reshape(ifft2c(reshape(x,N1,N2,Nc)),[],1),N1,N2,Nc,d1,d2);
Dh = @(x) reshape(fft2c(reshape(difference_H(x,N1,N2,Nc,d1,d2),N1,N2,Nc)),[],1);
DhD = reshape(real(Dh(D(ones(N1,N2,Nc)))),N1,N2,Nc);

%%
load('TV_brain_sigma8.mat')

