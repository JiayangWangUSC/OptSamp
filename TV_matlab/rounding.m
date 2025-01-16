%% efficient rounding

clear all;
close all;
clc;

SNR = 3;
%load(['./weight_snr',num2str(int8(SNR))]);
load(['./opt75_mse_mask_snr',num2str(int8(SNR))]);
total = round(sum(weight));
weight_int = round(weight);
fraction = weight_int - weight;
total_round = sum(weight_int);
if total_round > total
    k = total_round-total;
    [~,ind] = maxk(fraction,k);
    weight_int(ind) = weight_int(ind) - 1;
elseif total_round < total
    k = total - total_round;
    [~,ind] = maxk(-fraction,k);
    weight_int(ind) = weight_int(ind) + 1;
end

%save(['./weight_int_snr',num2str(int8(SNR))], 'weight_int')
save(['./opt75_mse_maskint_snr',num2str(int8(SNR))],'weight_int');