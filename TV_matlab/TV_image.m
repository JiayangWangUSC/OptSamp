clear all;
close all;
clc;

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1_demo/';
dirname = dir(datapath);

%% single image recon
N1 = 320; N2 = 320; Nc = 16; Ns =8;

%% difference matrix
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

%% define snr w. 8-averaging
factor = 8;
SNR = 3; % SNR after uniform averaging
sigma = sqrt(8)*0.12/SNR;
reso = 5;

%% reconstruction parameters initialization
if SNR == 2
    rho = 0.8; beta = 0.5;
elseif SNR == 3
    rho = 0.6; beta = 0.5;
elseif SNR == 5
    rho = 0.3; beta = 0.5;
elseif SNR == 10
    rho = 0.1; beta = 0.5;
end

% if SNR == 2
%     rho = 0.6; beta = 0.5;
% elseif SNR == 3
%     rho = 0.4; beta = 0.5;
% elseif SNR == 5
%     rho = 0.19; beta = 0.5;
% elseif SNR == 10
%     rho = 0.07; beta = 0.5;
% end

MaxIter = 20;

%% mask generation
weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * ones(1,N2-32*reso);
uni_mask = zeros(N1,N2);
uni_mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = repmat(weight,[N1-32*reso,1]);
uni_mask = repmat(uni_mask,[1,1,Nc]);

load(['./weight_snr',num2str(int8(SNR)),'_reso',num2str(int8(reso))])
opt_mask = zeros(N1,N2);
opt_mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = repmat(weight,[N1-32*reso,1]);
opt_mask = repmat(opt_mask,[1,1,Nc]);

%%
num_slice = 3;

disp([datapath,dirname(3).name]);
kspace = h5read([datapath,dirname(3).name],'/kspace_central');
Maps = h5read([datapath,dirname(3).name],'/sense_central');    

kData = complex(kspace(:,:,1:Nc,num_slice),kspace(:,:,Nc+1:2*Nc,num_slice));
maps = complex(Maps(:,:,1:Nc,num_slice),Maps(:,:,Nc+1:2*Nc,num_slice));
noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
gt = abs(sum(ifft2c(kData).*conj(maps),3));
support = sum(maps.*conj(maps),3);

%%
data = abs(sum(conj(maps).*ifft2c(kData+noise/sqrt(factor)),3));

%%
rho = 0.8; beta = 0.5;
recon_uni = TV(sqrt(uni_mask).*kData + (uni_mask>0).*noise,sqrt(uni_mask),rho,beta,MaxIter,D,Dh,DhD);
recon_uni = abs(sum(ifft2c(reshape(0.95*recon_uni,N1,N2,Nc)+0.05*kData).*conj(maps),3));
%imwrite(recon_uni/max(gt(:))*1.5,['/home/wjy/Project/optsamp_result/uni_tv_snr',num2str(SNR),'.png'])
%imwrite((abs(recon_uni-gt))/max(gt(:))*5,['/home/wjy/Project/optsamp_result/uni_tv_error_snr',num2str(SNR),'.png'])
norm(abs(recon_uni(:)-gt(:)))/norm(gt(:))
ssim(recon_uni/max(gt(:)),gt/max(gt(:)),'DynamicRange', 1)

%%
rho = 0.5; beta = 0.5;
recon_opt = TV(sqrt(opt_mask).*kData + (opt_mask>0).*noise,sqrt(opt_mask),rho,beta,MaxIter,D,Dh,DhD);
recon_opt = abs(sum(ifft2c(reshape(0.9*recon_opt,N1,N2,Nc)+0.1*kData).*conj(maps),3));

norm(abs(recon_opt(:)-gt(:)))/norm(gt(:))
ssim(recon_opt/max(gt(:)),gt/max(gt(:)),'DynamicRange', 1)

%imwrite((recon_opt)/max(gt(:))*1.5,['/home/wjy/Project/optsamp_result/opt_tv_snr',num2str(SNR),'.png'])
%imwrite((abs(recon_opt-gt))/max(gt(:))*5,['/home/wjy/Project/optsamp_result/opt_tv_error_snr',num2str(SNR),'.png'])

%%
function recon = TV(usData,kMask,rho,beta,MaxIter,D,Dh,DhD)
AhA = kMask.*kMask + rho*DhD;
AhA_dagger = AhA;
AhA_dagger(find(AhA)) = 1./AhA(find(AhA));
kMask_dagger = kMask;
kMask_dagger(find(kMask)) = 1./kMask(find(kMask));

x = usData.*kMask_dagger;
x = x(:);
z = threshold(D(x),beta);
u = 0*z;
fd = kMask(:).*usData(:);

for k = 1:MaxIter
    x = AhA_dagger(:).*(fd+rho*Dh(z-u));
    Dx = D(x);
    %disp(norm(Dx(:)));
    z = threshold(Dx+u,beta);
    u = u + Dx - z; 
    %norm(x-kData(:))
end
recon = x;
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
function result = difference(x,N1,N2,Nc,d1,d2)
x = reshape(x,N1,N2,Nc);
D1 = x;
D2 = x;
for c = 1:Nc
    D1(:,:,c) = d1*x(:,:,c);
    D2(:,:,c) = x(:,:,c)*d2;
end
result = [reshape(D1,[],1);reshape(D2,[],1)];
end

function result = difference_H(x,N1,N2,Nc,d1,d2)
D1 = reshape(x(1:end/2),N1,N2,Nc);
D2 = reshape(x(end/2+1:end),N1,N2,Nc);
result = zeros(N1,N2,Nc);
for c = 1:Nc
    result(:,:,c) = d1'*D1(:,:,c)+D2(:,:,c)*d2';
end
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
