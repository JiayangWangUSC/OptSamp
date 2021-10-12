clear all;
close all;
clc;

%% load data

datapath = '/home/wjy/Project/fastmri_dataset/test/';

dirname = dir(datapath);

[N1,N2, Nc] = size(kData);

fft2c = @(x) fftshift(fft2(ifftshift(x)));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)));

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

%% reconstruction parameters initialization
rho = 4;
beta = 1e-2;
MaxIter = 20;

factor = 8;
weight = factor*ones(N1,N2);
kMask = repmat(sqrt(weight),[1,1,Nc]);
usData = kMask.*kData+noise;
MaskLoop = 1;

%%
recon = TV(usData,kMask,rho,beta,MaxIter,D,Dh,DhD);
imr = ifft2c(reshape(recon,N1,N2,Nc));
ImR = sqrt(sum(abs(imr).^2,3));
figure;imagesc(ImR);colormap(gray);axis equal;axis tight; axis off; caxis([min(Im(:)),0.6*max(Im(:))]);

%%
for N = 1:MaskLoop
%% mask optimization
noiseLoop =1;
step = 1000;
Gradient = 0;
NRMSE = 0;
for L3 = 1:noiseLoop
noise = Noise(:,:,:,L3);     
usData = kMask.*kData + noise;
AhA = kMask.*kMask + rho*DhD;
AhA_dagger = AhA;
AhA_dagger(find(AhA)) = 1./AhA(find(AhA));

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
    %norm(x-kData(:))
end
recon = x;
imr = ifft2c(reshape(recon,N1,N2,Nc));
ImR = sqrt(sum(abs(imr).^2,3));
%NRMSE = NRMSE + norm(sp.*(ImR-Im))/norm(sp.*Im);
NRMSE = NRMSE + norm(ImR-Im)/norm(Im);

%figure;imagesc(ImR);colormap(gray);axis equal;axis tight; axis off; caxis([min(Im(:)),0.6*max(Im(:))]);
%%
Grad = 0;
%dx = sp.*((ImR-Im)./ImR);
dx = (ImR-Im)./ImR;
dx = fft2c(repmat(dx,[1,1,Nc]).*imr);
dx = dx(:);
%dx = recon-kData(:);
%dx = dx(:);
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

Gradient = Gradient + Grad;
end
%%
%du = - rho* D(AhA_dagger(:).*dx) + threshold_grad(D(X(:,k+1))+U(:,k),beta);
%%
%dx = Dh(du+threshold_grad(D(X(:,k))+U(:,k-1),beta).*dz);
% a = [1+2i,0.5-0.5i];
% threshold(a,1)
% threshold_grad(a,1)


%%

Gradient = reshape(Gradient,N1,N2,Nc);
Gradient = sum(real(Gradient)+imag(Gradient),3);
%Gradient = sum(Gradient,3);
Gradient = Gradient-mean(Gradient(:));
Gradient = Gradient/norm(Gradient(:));

weight = weight - step* Gradient;
for p = 1:10
    weight(weight<1) = 1;
    weight = weight - mean(weight(:)) + factor;
end
weight(weight<1) = 1;
kMask = repmat(sqrt(weight),[1,1,Nc]);
display(NRMSE);
end

%%
%weight =  struct2array(load('weight_whole'));
% weight = factor*ones(N1,N2);
% kMask = repmat(sqrt(weight),[1,1,Nc]);
% noise = Noise(:,:,:,1);     
% usData = kMask.*kData + noise;
% recon = TV(usData,kMask,rho,beta,MaxIter,D,Dh,DhD);
% %recon = usData./kMask;
% imr = ifft2c(reshape(recon,N1,N2,Nc));
% ImR = sqrt(sum(abs(imr).^2,3));
% 
% kerror = sum(abs(reshape(recon,N1,N2,Nc)-kData),3);
%%

% figure(1);imagesc(ImR);axis equal;axis off;colormap(gray);caxis([min(Im(:)),0.6*max(Im(:))]);
% figure(2);imagesc(abs(ImR-Im));axis equal;axis tight;axis off;colormap('jet');caxis([min(Im(:)),0.2*max(Im(:))]);colorbar;
% figure(3);imagesc(kerror);axis equal;axis tight;axis off;colormap('jet');caxis([0,0.1*max(abs(kData(:)))]);colorbar;
% figure(4);imagesc(weight);axis equal;axis tight;axis off;colorbar;
%%
function recon = TV(usData,kMask,rho,beta,MaxIter,D,Dh,DhD)
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