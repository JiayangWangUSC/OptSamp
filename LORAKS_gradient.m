clear all;
close all;
clc;

%% load data
load('6197JH_T2_slice30.mat');
kData = conj(kData);

%kData = kData(end/2-50:end/2+50,end/2-50:end/2+50,:);
[N1,N2, Nc] = size(kData);
kData = 1e5*kData;
%N1 = 20;
%N2 = 20;
%kData = kData(end/2-10:end/2+9,end/2-10:end/2+9,:);
%N2 = 186;
%kData = kData(:,1:N2,:);
%ACS = 86:101;
fft2c = @(x) fftshift(fft2(ifftshift(x)));
ifft2c = @(x) fftshift(ifft2(ifftshift(x)));
imf = ifft2c(kData);
Im = sqrt(sum(abs(imf).^2,3));
%figure(1);imagesc(Im);colormap(gray);axis equal;axis tight; axis off;drawnow;

%%
sp = zeros(N1,N2);
sp(Im>0.06*max(Im(:)))=1;


%% LORAKS operator
% P_M: LORAKS matrix constructor
% Ph_M: its adjoint
% mm: diagonal element of Ph_M(P_M) matrix
R = 3;
P_M = @(x) LORAKS_operators(x,N1,N2,Nc,R,1,[]);
Ph_M = @(x) LORAKS_operators(x,N1,N2,Nc,R,-1,[]);
mm = repmat(LORAKS_operators(LORAKS_operators(ones(N1*N2,1,'like',kData)...
            ,N1,N2,1,R,1,[]),N1,N2,1,R,-1,[]), [Nc 1]);

%mm_reg = max(mm(:));        
beta = 116;
mm_reg = beta+mm;
mm_reg(mm_reg>max(mm(:))) = max(mm(:));


%S = P_M(kData);
%[U,E] = svd_left(S, 1944);
        
        
rank = 45;

%% 
MaxIter = 5;
%tol = 1e-3;
lambda = 5;

factor = 8;
weight = factor*ones(N1,N2);
%weight = (factor-0.2)*ones(N1,N2);
%weight(end/2-10:end/2+10,end/2-10:end/2+10) = weight(end/2-10:end/2+10,end/2-10:end/2+10) + 0.2*N1*N2/21/21;
%weight =  struct2array(load('weight_loraks_n30_roi.mat'));
%weight = (factor-1)*ones(N1,N2);
%weight(:,end-49:end) = weight(:,end-49:end)+N2/50;
%%
% Noise = zeros(N1,N2,Nc,10);
 noise_level = 1*sqrt(factor)*1e-2*sqrt(0.5)*max(vect([real(kData),imag(kData)]));
% for n = 1:10
% Noise(:,:,:,n) = noise_level*randn(N1,N2,Nc) + 1i* noise_level*randn(N1,N2,Nc);
% end
load('Noise');
%% loop for weight optimization
Loop = 50;
step = 100;
for loop = 1:Loop
%noise = noise_level*randn(N1,N2,Nc) + 1i* noise_level*randn(N1,N2,Nc);
kMask = repmat(sqrt(weight),[1,1,Nc]);
%noise = 1.5*Noise(:,:,:,mod(loop-1,10)+1);
%noise = Noise(:,:,:,1);
noise = noise_level*randn(N1,N2,Nc) + 1i* noise_level*randn(N1,N2,Nc);
%noise = 1.5*noise(end/2-50:end/2+50,end/2-50:end/2+50,:);
noise = 3*noise;
usData = kMask.*kData+noise;
x = usData./kMask;
x = x(:);
AhA = kMask(:).*kMask(:) + lambda*mm_reg;
%AhA_dagger = AhA;
%AhA_dagger(find(AhA)) = 1./AhA(find(AhA));
AhA_dagger = 1./AhA;
%% LORAKS forward
fd = kMask(:).*usData(:);

W = [];
Q = [];
V = [];
K = [];

for iter = 1:MaxIter
    x_prev = x;
    MM = P_M(x);
    [pmm,S] = svd_left(MM,rank);
    MMr = Ph_M(pmm*pmm'*MM);
    x = AhA_dagger.*(fd + lambda*MMr);
    
    W(:,:,iter) = MM;
    Q(:,iter) = MMr;
    V(:,:,iter) = pmm;
    S = repmat(S,[1,rank])-repmat(S',[rank,1]);
    S(find(S)) = 1./(S(find(S)));
    K(:,:,iter) = S;
    
    %if norm(x-x_prev)/norm(x_prev)<tol
    %    display(iter);
    %    break;
    %end
    
end
recon = x;
RealIter = iter;

%% LORAKS backward
imr = ifft2c(reshape(recon,N1,N2,Nc));
ImR = sqrt(sum(abs(imr).^2,3));
%norm(ImR-Im)/norm(Im)
norm(sp.*(ImR-Im))/norm(sp.*Im)
%figure;imagesc(ImR);colormap(gray);axis equal;axis tight; axis off; caxis([min(Im(:)),0.6*max(Im(:))]);
Grad = 0;
%dBeta = 0;
df = sp.*((ImR-Im)./ImR);
%df = (ImR-Im)./ImR;
df = fft2c(repmat(df,[1,1,Nc]).*imr);
df = df(:);
for iter = RealIter:-1:1
    
        df_prev = df;
        dM = AhA_dagger.*AhA_dagger.*(2*lambda*kMask(:).*(mm_reg.*kData(:)-Q(:,iter))+(mm_reg-lambda*kMask(:).*kMask(:)).*noise(:));
        Grad = Grad + complex_odot(dM,df);
        
        %dBeta = dBeta + sum(sum(Q(:,iter-1)'*df_prev));
        
        dX = P_M(lambda*AhA_dagger.* df_prev);
        dH = dX*W(:,:,iter)';
        tempV = V(:,:,iter);
        dW1 = tempV*tempV'*dX;
        dV = (dH + dH')*tempV;
        dA = K(:,:,iter)'.*(tempV'*dV);
        dW2 = tempV*(dA+dA')*(tempV'*W(:,:,iter));
        df = Ph_M(dW1+dW2);
        
   if iter == 1
        dM = -noise./kMask./kMask;
        Grad = Grad + complex_odot(dM(:),df);
    end
end

Grad = Grad./kMask(:)/2;
Grad = reshape(Grad,N1,N2,Nc);
Grad = sum(real(Grad)+imag(Grad),3);
%Grad = sum(abs(Grad),3);
%Grad = 2*sigmoid((Grad-mean(Grad(:)))/std(Grad(:)))-1;
Grad = Grad-mean(Grad(:));
Grad = Grad/norm(Grad(:));

weight = weight - step * Grad;
for p = 1:10
    weight(weight<1) = 1;
    weight = weight - mean(weight(:)) + factor;
end
weight(weight<1) = 1;
%if mean(weight(:))>factor
%    break;
%end
end

%%
%g = complex_odot(dM,df);

%% 
function result = sigmoid(x)
    result = 1./(1+exp(-x));
end

%%
function result = even(int)
result = not(rem(int,2));
end

%%
function out = vect( in )
out = in(:);

end

%%
function [U,E] = svd_left(A, r)
% Left singular matrix of SVD (U matrix)
% parameters: matrix, rank (optional)
if nargin < 2
    [U,E] = eig(A*A'); % surprisingly, it turns out that this is generally faster than MATLAB's svd, svds, or eigs commands
    [~,idx] = sort(abs(diag(E)),'descend');
    U = U(:,idx);
    E = diag(E);
    E = E(idx);
else
    [U,E] = eig(A*A');
    [~,idx] = sort(abs(diag(E)),'descend');
    U = U(:,idx(1:r));
    E = diag(E);
    E = E(idx(1:r));
end
end

%%
function result = complex_odot(x,y)
    result = complex(real(x).*real(y),imag(x).*imag(y));
end


%% 
function result = LORAKS_operators(x, N1, N2, Nc, R, LORAKS_type, weights)
if LORAKS_type == 1     % S matrix
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros(patchSize,2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2,'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);

            tmp = x(Ind,:)-x(Indp,:);
            result(:,1,:,k) = real(tmp);
            result(:,2,:,k) = -imag(tmp);

            tmp = x(Ind,:)+x(Indp,:);
            result(:,1,:,k+end/2) = imag(tmp);
            result(:,2,:,k+end/2) = real(tmp);
        end
    end

    result = reshape(result, patchSize*Nc*2,(N1-2*R-even(N1))*(N2-2*R-even(N2))*2);

elseif LORAKS_type == -1 % S^H (Hermitian adjoint of of S matrix) 
    result = zeros(N1*N2,Nc,'like',x);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));

    x = reshape(x, [patchSize*2, Nc, (N1-2*R-even(N1))*(N2-2*R-even(N2))*2]);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            Ind = sub2ind([N1,N2],i+in1,j+in2);
            Indp = sub2ind([N1,N2],-i+in1+2*ceil((N1-1)/2)+2,-j+in2+2*ceil((N2-1)/2)+2);

            result(Ind,:) = result(Ind,:) + complex(x(1:patchSize,:,k) + x(patchSize+1:2*patchSize,:,nPatch+k), x(1:patchSize,:,nPatch+k) - x(patchSize+1:2*patchSize,:,k)); 
            result(Indp,:) = result(Indp,:) + complex( - x(1:patchSize,:,k) + x(patchSize+1:2*patchSize,:,nPatch+k), x(1:patchSize,:,nPatch+k) + x(patchSize+1:2*patchSize,:,k)); 

        end
    end

    result = vect(result);

elseif LORAKS_type == 2  % C matrix
    x = reshape(x,N1*N2,Nc);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);

    result = zeros(patchSize*Nc,(N1-2*R-even(N1))*(N2-2*R-even(N2)),'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(:,k) = vect(x(ind,:));
        end
    end

elseif LORAKS_type == -2 % C^H (Hermitian adjoint of of C matrix) 
    result = zeros(N1*N2,Nc,'like',x);

    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(ind,:) = result(ind,:)+ reshape(x(:,k),patchSize,Nc);
        end
    end
    result = vect(result);

elseif LORAKS_type == 3  % W matrix
    W1 = weights(:,:,1:Nc,1);
    W2 = weights(:,:,1:Nc,2);
    
    W1x = reshape(W1(:).*x(:),N1*N2,Nc);
    W2x = reshape(W2(:).*x(:),N1*N2,Nc);
    
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';

    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));
    result = zeros(patchSize*Nc,(N1-2*R-even(N1))*(N2-2*R-even(N2))*2,'like',x);

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result(:,k) = vect(W1x(ind,:));
            result(:,nPatch+k) = vect(W2x(ind,:));
        end
    end
    
elseif LORAKS_type == -3  % W^H (Hermitian adjoint of of W matrix)
    W1 = reshape(weights(:,:,1:Nc,1),[N1*N2,Nc]);
    W2 = reshape(weights(:,:,1:Nc,2),[N1*N2,Nc]);
    
    result1 = zeros(N1*N2,Nc,'like',x);
    result2 = zeros(N1*N2,Nc,'like',x);
    
    [in1,in2] = meshgrid(-R:R,-R:R);
    i = find(in1.^2+in2.^2<=R^2);

    in1 = in1(i)';
    in2 = in2(i)';
    patchSize = numel(in1);
    nPatch = (N1-2*R-even(N1))*(N2-2*R-even(N2));

    k = 0;
    for i = R+1+even(N1):N1-R
        for j = R+1+even(N2):N2-R
            k = k+1;
            ind = sub2ind([N1,N2],i+in1,j+in2);
            result1(ind,:) = result1(ind,:)+ reshape(x(:,k),patchSize,Nc);
            result2(ind,:) = result2(ind,:)+ reshape(x(:,nPatch+k),patchSize,Nc);
        end
    end
    result = vect(conj(W1).*result1 + conj(W2).*result2);
end
end

