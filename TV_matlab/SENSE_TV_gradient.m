clear all;
close all;
clc;

addpath(genpath('./SPIRiT_v0.3'));
%% load data

datapath = '/home/wjy/Project/fastmri_dataset/brain_copy/';
dirname = dir(datapath);
N1 = 384; N2 = 396; Nc = 16;

subject = [];
slice = [];

for i = 3:length(dirname)
    fname = dirname(i).name;
    kspace = h5read([datapath,fname],'/kspace');
    kspace = complex(kspace.r,kspace.i);
    kspace = permute(kspace,[4,2,1,3]);
    for snum = 1:size(kspace,1)
        subject = [subject;fname];
        slice = [slice;snum];
    end
end

datalen = length(slice);
batch_size = 8;
batch_num = datalen/batch_size;

%%
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

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

%% reconstruction parameters initialization
sigma = 0.4;
noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));

factor = 8;
weight = factor*ones(1,N2);

rho = 8;
lambda = 0.5;
beta = 1; 
MaxIter = 10;

%%
kspace = h5read([datapath,dirname(3).name],'/kspace');
kspace = complex(kspace.r,kspace.i);
kspace = permute(kspace,[4,2,1,3]);
kData = undersample(reshape(kspace(3,:,:,:),2*N1,N2,Nc))/1e-4;

%%
kMask = repmat(sqrt(weight),[N1,1,Nc]);
kMask_dagger = kMask;
kMask_dagger(find(kMask)) = 1./(kMask(find(kMask)));

usData = kMask.*kData+noise;
maps = getmap(kData);

%% 
[N1,N2,Nc] = size(usData);
D = @(x) [reshape(d1*reshape(x,N1,N2),[],1);reshape(reshape(x,N1,N2)*d2,[],1)];
Dh = @(x) reshape(d1'*reshape(x(1:end/2),N1,N2) + reshape(x(end/2+1:end),N1,N2)*d2',[],1);
DhD = real(fft2c(reshape(Dh(D(ifft2c(ones(N1,N2)))),N1,N2)));
BhB_dagger = @(x) reshape(ifft2c(fft2c(reshape(x,N1,N2))./(rho+beta*DhD)),[],1);

FEh = @(x)reshape(sum(conj(maps).*ifft2c(reshape(x,N1,N2,Nc)),3),[],1);
FE = @(x) reshape(fft2c(maps.*repmat(reshape(x,N1,N2),[1,1,Nc])),[],1);
%%
AhA = kMask.*kMask + rho;
AhA_dagger = 1./AhA;

% initialization
k = usData(:).*kMask_dagger(:);
f = FEh(k);
u = 0 *k;
z = threshold(D(f),lambda/beta);
v = D(f) - z;
mk = usData(:).*kMask(:);
%%
for iter = 1:10
    k = AhA_dagger(:).*(mk+ rho*(FE(f)+u));
    f = BhB_dagger(rho*FEh(k-u)+beta*Dh(z-v));
    u = FE(f) - k + u;
    z = threshold(D(f)+v,lambda/beta);
    v = D(f) - z + v;
    
    norm(abs(f-Im(:)))/norm(abs(Im(:)))
    %norm(abs(D(f) - z))
end
recon = f;
%%
mean(abs(D(Im)))
%%
Im = FEh(kData);
image_norm(abs(FEh(k)-f))
image_norm(abs(FEh(k)-FEh(kData)))/image_norm(abs(FEh(kData)))

%%
function maps = getmap(kData)
[sx,sy,Nc] = size(kData);
ncalib = 24; % use 24 calibration lines to compute compression
ksize = [6,6]; % kernel size

% Threshold for picking singular vercors of the calibration matrix
% (relative to largest singlular value.

eigThresh_1 = 0.02;

% threshold of eigen vector decomposition in image space.
eigThresh_2 = 0.95;

% crop a calibration area
calib = crop(kData,[ncalib,ncalib,Nc]);

% compute Calibration matrix, perform 1st SVD and convert singular vectors
% into k-space kernels

[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));

[M,W] = kernelEig(k(:,:,:,1:idx),[sx,sy]);
maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);
end

%% threshold
function result = threshold(x,th)

    v = abs(x)-th;
    v(v<0) = 0;
    result = sign(x).*v;
end

function result = threshold_grad(x,th)
    rx = real(x);
    ix = imag(x);
    v = ones(size(rx));
    v((abs(x)-th)<0) = 0;
    grad_real = 1 - ix.^2./abs(x).^3*th;
    grad_imag = 1 - rx.^2./abs(x).^3*th;
    result = complex(grad_real,grad_imag).*v;
end

function result = complex_odot(x,y)
    result = complex(real(x).*real(y),imag(x).*imag(y));
end



%% 
function kspace = undersample(kspace)
    fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
    ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4;   
    im = ifft2c(kspace);
    im = im(192:575,:,:);
    kspace = fft2c(im);
end

%%
function value = image_norm(image)
    value = norm(image(:));
end
%%
function [kernel, S] = dat2Kernel(data, kSize)
% kernel = dat2Kernel(data, kSize,thresh)
%
% Function to perform k-space calibration step for ESPIRiT and create
% k-space kernels. Only works for 2D multi-coil images for now.  
% 
% Inputs: 
%       data - calibration data [kx,ky,coils]
%       kSize - size of kernel (for example kSize=[6,6])
%
% Outputs: 
%       kernel - k-space kernels matrix (not cropped), which correspond to
%                the basis vectors of overlapping blocks in k-space
%       S      - (Optional parameter) The singular vectors of the
%                 calibration matrix
%
%
% See also:
%           kernelEig
%
% (c) Michael Lustig 2013



[sx,sy,nc] = size(data);
imSize = [sx,sy] ;

tmp = im2row(data,kSize); [tsx,tsy,tsz] = size(tmp);
A = reshape(tmp,tsx,tsy*tsz);

[U,S,V] = svd(A,'econ');
    
kernel = reshape(V,kSize(1),kSize(2),nc,size(V,2));
S = diag(S);S = S(:);
end