clear all;
close all;
clc;

addpath(genpath('./SPIRiT_v0.3'));
%% load data

datapath = '/home/wjy/Project/fastmri_dataset/test/';
%datapath = '/project/jhaldar_118/jiayangw/OptSamp/dataset/val/';
dirname = dir(datapath);
%data = h5read('file_brain_AXT2_200_6002217.h5','/home/wjy/Project/fastmri_dataset/test');
%kspace = h5read([datapath,dirname(3).name],'/kspace');
%kspace = complex(kspace.r,kspace.i);
%kspace = permute(kspace,[4,2,1,3]);
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
sigma = 0.3;
noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
factor = 8;
weight = factor*ones(1,N2);
rho = 1;
beta = 0.1; % (noise_level,rho,beta): (0.5, 1, 0.5),(1, 2, 1),(0.3,0.6,0.5)

MaxIter = 10;
%%
%load('/home/wjy/TV_noise05_mask.mat');
kspace = h5read([datapath,dirname(3).name],'/kspace');
kspace = complex(kspace.r,kspace.i);
kspace = permute(kspace,[4,2,1,3]);
kData = undersample(reshape(kspace(3,:,:,:),2*N1,N2,Nc))/1e-4;
kMask = repmat(sqrt(weight),[N1,1,Nc]);
usData = kMask.*kData+noise;
recon = TV(usData,kMask,rho,beta,MaxIter,d1,d2);
imr = ifft2c(reshape(recon,N1,N2,Nc));
ImR = sqrt(sum(abs(imr).^2,3));
Im = sqrt(sum(abs(ifft2c(reshape(kData,N1,N2,Nc))).^2,3));
ImN= sqrt(sum(abs(ifft2c(reshape(usData./kMask,N1,N2,Nc))).^2,3));
support = zeros(N1,N2);
support(Im>0.06*max(Im(:))) = 1;

%%
%patch = ImN(221:260,101:150);
%patch = imresize(patch,[80,80],'nearest');
%imwrite(patch/max(Im(:))*2,'/home/wjy/Project/OptSamp/result_local/TV_patch1_noise08.png');
maps = getmap(usData./kMask);
%%
 image_norm(support.*(ImN-Im))/image_norm(support.*Im)
 image_norm(support.*(ImR-Im))/image_norm(support.*Im)
%%
%ssim(support.*ImR/max(Im(:))*256,support.*Im/max(Im(:))*256)

%%
epoch_max = 30;
step = 10;
train_loss = zeros(1,epoch_max);
for epoch = 1:epoch_max
    disp(epoch);
    loss = 0;
    for batch = 1:batch_num
        kMask = repmat(sqrt(weight),[N1,1,Nc]);
        AhA = kMask.*kMask + rho*DhD;
        AhA_dagger = AhA;
        AhA_dagger(find(AhA)) = 1./AhA(find(AhA));
        Gradient = zeros(N1,N2,batch_size);
        mse = zeros(1,batch_size);
        parfor (datanum = 1:batch_size, batch_size)
            % data load
            fname = subject(batch_size*(batch-1)+datanum,:);
            slicenum = slice(batch_size*(batch-1)+datanum);
            kspace = h5read([datapath,fname],'/kspace');
            kspace = complex(kspace.r,kspace.i);
            kspace = permute(kspace,[4,2,1,3]);
            kData = undersample(reshape(kspace(slicenum,:,:,:),2*N1,N2,Nc))/1e-4;
            
            % sample
            noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
            usData = kMask.*kData + noise;
        
            %recon
            x = usData./kMask;
            x = x(:);
            X = repmat(0*x,[1,MaxIter]);
            z = threshold(D(x),beta);
            Z = repmat(0*z,[1,MaxIter]);
            u = 0*z;
            U = repmat(0*u,[1,MaxIter]);
            fd = kMask(:).*usData(:);

            for k = 1:MaxIter
                X(:,k) = x;
                Z(:,k) = z;
                U(:,k) = u;
                x = AhA_dagger(:).*(fd+rho*Dh(z-u));
                Dx = D(x);
                z = threshold(Dx+u,beta);
                u = u + Dx - z; 
            end
            recon = x;
            imr = ifft2c(reshape(recon,N1,N2,Nc));
            ImR = sqrt(sum(abs(imr).^2,3));
            Im = sqrt(sum(abs(ifft2c(reshape(kData,N1,N2,Nc))).^2,3));
            
            support = zeros(N1,N2);
            support(Im>0.06*max(Im(:))) = 1;
            
            %% backward propagation
            Grad = 0;
            dx = support.*((ImR-Im)./Im);
            dx = fft2c(repmat(dx,[1,1,Nc]).*imr);
            dx = dx(:);
            for k = MaxIter:-1:1
                dW = AhA_dagger.*AhA_dagger.*(rho*DhD.*(kData+noise./kMask/2)-kMask.*noise/2-rho*reshape(Dh(Z(:,k)-U(:,k)),N1,N2,Nc));
                Grad = Grad + complex_odot(dW(:),dx);
                if k == MaxIter
                    du = - rho* D(AhA_dagger(:).*dx);
                else
                    du = du - rho* D(AhA_dagger(:).*dx) + complex_odot(threshold_grad(D(X(:,k+1))+U(:,k),beta),dz);
                end
                    dz = rho* D(AhA_dagger(:).*dx) - du;
    
                if k == 1
                    dx = Dh(complex_odot(threshold_grad(D(X(:,k)),beta),dz));
                    dW = -noise(:)./(kMask(:)).^3/2;
                    Grad = Grad + complex_odot(dW,dx);
                else
                    dx = Dh(du + complex_odot(threshold_grad(D(X(:,k))+U(:,k-1),beta),dz));
                end
            end
            Grad = reshape(Grad,N1,N2,Nc);
            Grad = sum(real(Grad)+imag(Grad),3);
            Gradient(:,:,datanum) = Grad;
            
            mse(datanum) = image_norm(support.*(ImR-Im))^2;
    
        end
        
        Gradient = sum(sum(Gradient,3),1);
        Gradient = Gradient-mean(Gradient(:));
        Gradient = Gradient/norm(Gradient(:));

        weight = weight - step* Gradient;
        for p = 1:10
            weight(weight<1) = 1;
            weight = weight - mean(weight(:)) + factor;
        end
        weight(weight<1) = 1;
        if mod(batch,100) == 0
            disp(['epoch:',num2str(epoch),' batch:',num2str(batch),' train loss:',num2str(mean(mse))]);
        end
        loss = loss + mean(mse);
    end
    train_loss(epoch) = loss/batch_num;
end

%save TV_noise08_train_loss train_loss
save ./result_local/TV_noise03_mask.mat weight


%%
function recon = TV(usData,kMask,rho,beta,MaxIter,d1,d2)

maps = getmap(usData./kMask);

[N1,N2,Nc] = size(usData);
D = @(x) difference(reshape(sum(conj(maps).*ifft2c(reshape(x,N1,N2,Nc)),3),[],1),N1,N2,d1,d2);
Dh = @(x) reshape(fft2c(repmat(reshape(difference_H(x,N1,N2,d1,d2),N1,N2),[1,1,Nc]).*maps),[],1);
DhD = reshape(Dh(D(ones(N1,N2,Nc))),N1,N2,Nc);

AhA = kMask.*kMask + rho*DhD;
AhA_dagger = AhA;
AhA_dagger(find(AhA)) = 1./AhA(find(AhA));

x = usData./kMask;
x = x(:);
z = threshold(D(x),beta);
u = 0*z;
fd = kMask(:).*usData(:);

for k = 1:MaxIter
    x = AhA_dagger(:).*(fd+rho*Dh(z-u));
    Dx = D(x);
    
    z = threshold(Dx+u,beta);
    u = u + Dx - z; 
    %norm(x-kData(:))
end
recon = x;
end
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
    %v = abs(real(x))-th;
    %v(v<0)=0;
    %u = abs(imag(x))-th;
    %u(u<0)=0;
    %result= complex(sign(real(x)).*u,sign(imag(x)).*v);
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
function result = difference(x,N1,N2,d1,d2)
x = reshape(x,N1,N2);
D1 = d1*x;
D2 = x*d2;
result = [reshape(D1,[],1);reshape(D2,[],1)];
end

function result = difference_H(x,N1,N2,d1,d2)
D1 = reshape(x(1:end/2),N1,N2);
D2 = reshape(x(end/2+1:end),N1,N2);
result = d1'*D1+D2*d2';
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