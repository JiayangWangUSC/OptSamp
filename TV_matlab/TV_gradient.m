clear all;
close all;
clc;

%% define snr w. 8-averaging
factor = 8;
SNR = 2; % SNR after uniform averaging

sigma = 0.12*sqrt(8)/SNR;  

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4; 

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1/';
dirname = dir(datapath);

%% difference matrix
N1 = 320; N2 = 320; Nc = 16; Ns =8;

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
if SNR == 2
    rho = 0.8; beta = 0.5;
elseif SNR == 3
    rho = 0.6; beta = 0.5;
elseif SNR == 5
    rho = 0.3; beta = 0.5;
elseif SNR == 10
    rho = 0.1; beta = 0.5;
end

MaxIter = 20;
%%
for reso = 0:9

epoch_max = 2;
step = 0.1;
train_loss = zeros(1,epoch_max);

%weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * ones(1,N2-32*reso);
load(['./weight_snr',num2str(int8(3)),'_reso',num2str(int8(reso))])

for epoch = 1:epoch_max
    disp(epoch);
    step = step * 0.9;
    loss = 0;
    for sub_num = 3:length(dirname)
        kspace = h5read([datapath,dirname(sub_num).name],'/kspace_central');
        Maps = h5read([datapath,dirname(sub_num).name],'/sense_central');
        
       % Gradient = zeros(N1,N2,Ns);
        l2loss = zeros(1,Ns);
        
        for slice_num = 1:Ns
            % operator for updated mask
            kMask = zeros(N1,N2);
            kMask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = repmat(sqrt(weight),[N1-32*reso,1]);
            kMask = repmat(kMask,[1,1,Nc]);
            kMask_dagger = kMask;
            kMask_dagger(find(kMask)) = 1./kMask(find(kMask));
            AhA = kMask.*kMask + rho*DhD;
            AhA_dagger = AhA;
            AhA_dagger(find(AhA)) = 1./AhA(find(AhA));
            
            % load slice
            kData = complex(kspace(:,:,1:Nc,slice_num),kspace(:,:,Nc+1:2*Nc,slice_num));
            maps = complex(Maps(:,:,1:Nc,slice_num),Maps(:,:,Nc+1:2*Nc,slice_num));
            
            % generate noisy acquisition
            noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
            usData = kMask.*kData + noise;
        
            %recon
            x = usData.*kMask_dagger;
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
            ImR = sum(imr.*conj(maps),3);
            Im = sum(ifft2c(kData).*conj(maps),3);
            
            %% backward propagation
            Grad = 0;
            dx = fft2c(repmat(2*(ImR-Im),[1,1,Nc]).*maps);
            dx = dx(:);
            for k = MaxIter:-1:1
                dW = AhA_dagger.*AhA_dagger.*(rho*DhD.*(kData+noise.*kMask_dagger/2)-kMask.*noise/2-rho*reshape(Dh(Z(:,k)-U(:,k)),N1,N2,Nc));
                Grad = Grad + complex_odot(dW(:),dx);
                if k == MaxIter
                    du = - rho* D(AhA_dagger(:).*dx);
                else
                    du = du - rho* D(AhA_dagger(:).*dx) + complex_odot(threshold_grad(D(X(:,k+1))+U(:,k),beta),dz);
                end
                    dz = rho* D(AhA_dagger(:).*dx) - du;
    
                if k == 1
                    dx = Dh(complex_odot(threshold_grad(D(X(:,k)),beta),dz));
                    dW = -noise(:).*(kMask_dagger(:)).^3/2;
                    Grad = Grad + complex_odot(dW,dx);
                else
                    dx = Dh(du + complex_odot(threshold_grad(D(X(:,k))+U(:,k-1),beta),dz));
                end
            end
            
            Grad = reshape(Grad,N1,N2,Nc);
            Grad = sum(sum(real(Grad)+imag(Grad),3),1);
            
            Grad = Grad((16*reso+1):(N2-16*reso));
            Grad = Grad-mean(Grad(:));
            Grad = Grad/norm(Grad(:));
           
            weight = weight - step * Grad;
            weight = weight - 1;
            total = sum(weight);
            weight(weight<0) = 0;
            weight = weight/sum(weight)*total;
            weight = weight + 1;

            l2loss(slice_num) = sum(abs(ImR(:)-Im(:)).^2);
    
        end
        
        disp(['epoch:',num2str(epoch),' subject:',num2str(sub_num-2),' L2 loss:',num2str(mean(l2loss))]);
        
        loss = loss + mean(l2loss);
    end
    train_loss(epoch) = loss/(length(dirname)-2);
    
    save(['./weight_snr',num2str(int8(SNR)),'_reso',num2str(int8(reso))], 'weight')
end

end

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