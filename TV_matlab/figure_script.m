%% figure 
clear all;
close all;
clc;
N1 = 320; N2 = 320; Nc = 20; Ns =8;
%% plot averaging weights for SENSE-TV
load('weight_snr3.mat');
weight_snr3 = weight;
load('weight_snr5.mat');
weight_snr5 = weight;
load('weight_snr10.mat');
weight_snr10 = weight;
load('weight_int_snr3.mat');
weight_int_snr3 = weight_int;
load('weight_int_snr5.mat');
weight_int_snr5 = weight_int;
load('weight_int_snr10.mat');
weight_int_snr10 = weight_int;
uni_mask = 8 * ones(1,N2);
low50_mask = zeros(1,N2);
low50_mask(81:240) = 2 * 8;
low25_mask = zeros(1,N2);
low25_mask(121:200) = 4 * 8;

figure(1);
plot(uni_mask); hold on;
plot(low50_mask); hold on;
plot(low25_mask); hold on;
%plot(weight_snr3,'g--'); hold on;
plot(weight_int_snr3,'g'); hold on;
%plot(weight_snr5,'b--'); hold on;
plot(weight_int_snr5,'b'); hold on;
%plot(weight_snr10,'r--'); hold on;
plot(weight_int_snr10,'r'); hold on;
legend('Uniform','50% Low Frequency', '25% Low Frequency','SNR=3','SNR=5 ','SNR=10 ');
title('Averaging Pattern for SENSE-TV Recon')

%% plot averaging weights for UNet MSE;
load('opt_mse_mask_snr3.mat');
weight_snr3 = weight;
load('opt_mse_mask_snr5.mat');
weight_snr5 = weight;
load('opt_mse_mask_snr10.mat');
weight_snr10 = weight;
load('opt_mse_maskint_snr3.mat');
weight_int_snr3 = weight_int;
load('opt_mse_maskint_snr5.mat');
weight_int_snr5 = weight_int;
load('opt_mse_maskint_snr10.mat');
weight_int_snr10 = weight_int;
uni_mask = 8 * ones(1,N2);
low50_mask = zeros(1,N2);
low50_mask(81:240) = 2 * 8;
low25_mask = zeros(1,N2);
low25_mask(121:200) = 4 * 8;

figure(1);
plot(uni_mask); hold on;
plot(low50_mask); hold on;
plot(low25_mask); hold on;
plot(weight_snr3,'g--'); hold on;
plot(weight_int_snr3,'g'); hold on;
plot(weight_snr5,'b--'); hold on;
plot(weight_int_snr5,'b'); hold on;
plot(weight_snr10,'r--'); hold on;
plot(weight_int_snr10,'r'); hold on;
legend('Uniform','50% Low Frequency', '25% Low Frequency','SNR=3 Continuous','SNR=3','SNR=5 Continuous','SNR=5','SNR=10 Continuous','SNR=10');
title('Averaging Pattern for L2-Trained U-Net ')

%% plot averaging weights for UNet MAE
load('opt_mae_mask_snr3.mat');
weight_snr3 = weight;
load('opt_mae_mask_snr5.mat');
weight_snr5 = weight;
load('opt_mae_mask_snr10.mat');
weight_snr10 = weight;
load('opt_mae_maskint_snr3.mat');
weight_int_snr3 = weight_int;
load('opt_mae_maskint_snr5.mat');
weight_int_snr5 = weight_int;
load('opt_mae_maskint_snr10.mat');
weight_int_snr10 = weight_int;
uni_mask = 8 * ones(1,N2);
low50_mask = zeros(1,N2);
low50_mask(81:240) = 2 * 8;
low25_mask = zeros(1,N2);
low25_mask(121:200) = 4 * 8;

figure(1);
plot(uni_mask); hold on;
plot(low50_mask); hold on;
plot(low25_mask); hold on;
plot(weight_snr3,'g--'); hold on;
plot(weight_int_snr3,'g'); hold on;
plot(weight_snr5,'b--'); hold on;
plot(weight_int_snr5,'b'); hold on;
plot(weight_snr10,'r--'); hold on;
plot(weight_int_snr10,'r'); hold on;
legend('Uniform','50% Low Frequency', '25% Low Frequency','SNR=3 Continuous','SNR=3','SNR=5 Continuous','SNR=5','SNR=10 Continuous','SNR=10');
title('Averaging Pattern for L1-Trained U-Net ')