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
kspace = h5read([datapath,dirname(3).name],'/kspace_central');
kspace = complex(kspace(:,:,1:Nc,1),kspace(:,:,Nc+1:2*Nc,1));
maps = h5read([datapath,dirname(3).name],'/sense_central');
maps = complex(maps(:,:,1:Nc,1),maps(:,:,Nc+1:2*Nc,1));

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
sigma = sqrt(8)*0.15/SNR; % 

%% mask generation
uni_mask = factor * ones(N1,N2,Nc);
low50_mask = zeros(N1,N2,Nc);
low50_mask(:,81:240,:) = 2 * factor;
low25_mask = zeros(N1,N2,Nc);
low25_mask(:,121:200,:) = 4 * factor;
%load(['./weight_snr',num2str(int8(SNR))]);
%opt_mask = repmat(weight,N1,1,Nc);

%% recon parameters
if SNR == 3
    rho_uni = 0.6; beta_uni = 0.5;
    rho_low50 = 0.6; beta_low50 = 0.5;
    rho_low25 = 0.06; beta_low25 = 0.001;
elseif SNR == 5
    rho_uni = 0.3; beta_uni = 0.5;
    rho_low50 = 0.3; beta_low50 = 0.5;
    rho_low25 = 0.03; beta_low25 = 0.001;
elseif SNR ==10
    rho_uni = 0.1; beta_uni = 0.5;
    rho_low50 = 0.1; beta_low50 = 0.5;
    rho_low25 = 0.01; beta_low25 = 0.001;
end

MaxIter = 20;

%% acquisition
% image_gt = sum(ifft2c(kspace).*conj(maps),3);
% noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
% kMask = sqrt(uni_mask);
% kMask_dagger = kMask;
% kMask_dagger(find(kMask)) = 1./kMask(find(kMask));
% kspace_noise = kMask.* kspace + noise;

%imwrite(abs(sum(ifft2c(kMask_dagger.*kspace_noise).*conj(maps),3))/150,'low50_noise_snr10.png');

%% recon
% snr3: uni(rho=1.5,beta=50), low50(rho=1.5,beta=50), low25(rho=1.2,beta=50);
% snr5: uni(rho=0.4,beta=100), low50(rho=0.3,beta=100),low25(rho=1,beta=10);
% snr10: uni(rho=1,beta=10), low50(rho=0.8,beta=15),low25(rho=5,beta=3);

%rho = 0.8;
%beta = 15;

%MaxIter = 20;
%recon = TV(kspace_noise,kMask,rho,beta,MaxIter,D,Dh,DhD);

%im_recon = sum(ifft2c(reshape(recon,N1,N2,Nc)).*conj(maps),3);
%imwrite(abs(im_recon)/150,'low50_recon_snr10.png')

%norm(im_recon(:)-image_gt(:))/norm(image_gt(:))

%% 
MaxIter = 20;
rho_low25 = 0.06; beta_low25 = 0.001;

count = 0;
ssim_uni = 0; ssim_low50 = 0; ssim_low25 = 0; ssim_opt = 0;
nrmse_uni = 0; nrmse_low50 = 0; nrmse_low25 = 0; nrmse_opt = 0;
nmae_uni = 0; nmae_low50 = 0; nmae_low25 = 0; nmae_opt = 0;

for sub_num = 3%:length(dirname)
    disp([datapath,dirname(sub_num).name]);
    kspace = h5read([datapath,dirname(sub_num).name],'/kspace_central');
    Maps = h5read([datapath,dirname(sub_num).name],'/sense_central');    

    for slice_num = 1%:Ns
    count = count + 1;
    
    kData = complex(kspace(:,:,1:Nc,slice_num),kspace(:,:,Nc+1:2*Nc,slice_num));
    maps = complex(Maps(:,:,1:Nc,slice_num),Maps(:,:,Nc+1:2*Nc,slice_num));
    noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
    gt = abs(sum(ifft2c(kData).*conj(maps),3));
    pixelscale = max(gt(:));
    l2scale = norm(gt(:));
    l1scale = sum(gt(:));
    
    % recon_uni = TV(sqrt(uni_mask).*kData + noise,sqrt(uni_mask),rho_uni,beta_uni,MaxIter,D,Dh,DhD);
    % recon_uni = abs(sum(ifft2c(reshape(recon_uni,N1,N2,Nc)).*conj(maps),3));
    % ssim_uni = ssim_uni + ssim(recon_uni/pixelscale,gt/pixelscale,'DynamicRange', 1);
    % nrmse_uni = nrmse_uni + norm(recon_uni(:)-gt(:))/l2scale;
    % nmae_uni = nmae_uni + sum(abs(recon_uni(:)-gt(:)))/l1scale;
    
    % recon_opt = TV(sqrt(opt_mask).*kData + noise,sqrt(opt_mask),rho_uni,beta_uni,MaxIter,D,Dh,DhD);
    % recon_opt = abs(sum(ifft2c(reshape(recon_opt,N1,N2,Nc)).*conj(maps),3));
    % ssim_opt = ssim_opt + ssim(recon_opt/pixelscale,gt/pixelscale,'DynamicRange', 1);
    % nrmse_opt = nrmse_opt + norm(recon_opt(:)-gt(:))/l2scale;
    % nmae_opt = nmae_opt + sum(abs(recon_opt(:)-gt(:)))/l1scale;

    % recon_low50 = TV(sqrt(low50_mask).*kData + noise,sqrt(low50_mask),rho_low50,beta_low50,MaxIter,D,Dh,DhD);
    % recon_low50 = abs(sum(ifft2c(reshape(recon_low50,N1,N2,Nc)).*conj(maps),3));
    % ssim_low50 = ssim_low50 + ssim(recon_low50/pixelscale,gt/pixelscale,'DynamicRange', 1);
    % nrmse_low50 = nrmse_low50 + norm(recon_low50(:)-gt(:))/l2scale;
    % nmae_low50 = nmae_low50 + sum(abs(recon_low50(:)-gt(:)))/l1scale;

    recon_low25 = TV(sqrt(low25_mask).*kData + noise,sqrt(low25_mask),rho_low25,beta_low25,MaxIter,D,Dh,DhD);
    recon_low25 = abs(sum(ifft2c(reshape(recon_low25,N1,N2,Nc)).*conj(maps),3));
    ssim_low25 = ssim_low25 + ssim(recon_low25/pixelscale,gt/pixelscale,'DynamicRange', 1);
    nrmse_low25 = nrmse_low25 + norm(recon_low25(:)-gt(:))/l2scale;
    nmae_low25 = nmae_low25 + sum(abs(recon_low25(:)-gt(:)))/l1scale;

    end

    disp([ssim_uni/count,nrmse_uni/count,nmae_uni/count;ssim_low50/count,nrmse_low50/count,nmae_low50/count;ssim_low25/count,nrmse_low25/count,nmae_low25/count;ssim_opt/count,nrmse_opt/count,nmae_opt/count]);

end

       

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
