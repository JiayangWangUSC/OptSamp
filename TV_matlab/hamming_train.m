clear all;
close all;
clc;

%% load data
fft2c = @(x) fftshift(fft2(ifftshift(x)))/sqrt(size(x(:),1))*4;
ifft2c = @(x) fftshift(ifft2(ifftshift(x)))*sqrt(size(x(:),1))/4;

datapath = '/home/wjy/Project/fastmri_dataset/brain_T1/';
dirname = dir(datapath);

dict = [17
11
7
5
3
2
1
0.3];


%% parameters
N1 = 320; N2 = 320; Nc = 16; Ns =8;
factor = 8;
SNR = 2;
sigma = sqrt(8)*0.12/SNR;  

for reso = 0:7

beta = dict(reso+1);  % Higher beta for stronger noise suppression 
N1_wd = N1 - 32*reso;
N2_wd = N2 - 32*reso;
window = kaiser(N1_wd, beta) * kaiser(N2_wd, beta)';

%%
weight = factor*N1/(N1-32*reso)*N2/(N2-32*reso) * ones(1,N2-32*reso);
load(['./fftweight_snr',num2str(int8(5)),'_reso',num2str(int8(reso))]);

mask = zeros(N1,N2);
mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = window;
mask = repmat(mask,[1,1,Nc]);

%%
count = 0;
step = 0.1;
epoch_max = 10;

for epoch = 1:epoch_max
disp(epoch);
step = step * 0.9;
for sub_num = 3:length(dirname)
        kspace = h5read([datapath,dirname(sub_num).name],'/kspace_central');
        Maps = h5read([datapath,dirname(sub_num).name],'/sense_central');

        for slice_num = 1:Ns

        avg_mask = zeros(N1,N2);
        avg_mask((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso)) = repmat(weight,[N1-32*reso,1]);
        avg_mask = repmat(avg_mask,[1,1,Nc]);
        avg_mask_dagger = avg_mask;
        avg_mask_dagger(avg_mask>0) = 1./avg_mask(avg_mask>0);    
        
        count = count + 1;
        kData = complex(kspace(:,:,1:Nc,slice_num),kspace(:,:,Nc+1:2*Nc,slice_num));
        maps = complex(Maps(:,:,1:Nc,slice_num),Maps(:,:,Nc+1:2*Nc,slice_num));
        gt = abs(sum(ifft2c(kData).*conj(maps),3));
        pixelscale = max(gt(:));
        l2scale = norm(gt(:));

        noise = complex(sigma*randn(N1,N2,Nc),sigma*randn(N1,N2,Nc));
        recon = abs(sum(ifft2c(mask.*(kData+sqrt(avg_mask_dagger).*noise)).*conj(maps),3));

        grad = (ifft2c(repmat(2*(recon-gt),[1,1,Nc]).*conj(maps)).*mask).*noise;
        grad = complex_odot(grad,(-0.5*sqrt(avg_mask_dagger).^3));
        grad = grad((16*reso+1):(N1-16*reso),(16*reso+1):(N2-16*reso),:);
        
        grad = sum(sum(real(grad) + imag(grad),3),1);
        grad = grad-mean(grad(:));
        grad = grad/norm(grad(:));
        
        weight = weight - step * grad;
        weight = weight - 1;
        total = sum(weight);
        weight(weight<0) = 0;
        weight = weight/sum(weight)*total;
        weight = weight + 1;

        end
end
    save(['./fftweight_snr',num2str(int8(SNR)),'_reso',num2str(int8(reso))], 'weight')
end

end


function result = complex_odot(x,y)
    result = complex(real(x).*real(y),imag(x).*imag(y));
end

