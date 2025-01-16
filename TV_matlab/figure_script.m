%% figure 
clear all;
close all;
clc;
N1 = 320; N2 = 320; Nc = 20; Ns =8;

%% plot NRMSE-resolution trend
ssim_snr10 = [0.9409	0.9719	0.9747	0.9752	0.9766;
0.9542	0.9725	0.9747	0.9755	0.9764;
0.9638	0.9729	0.9736	0.9755	0.9755;
0.969	0.9709	0.9716	0.9742	0.9738;
0.9671	0.9665	0.9671	0.9701	0.9698;
0.9608	0.9594	0.9599	0.9632	0.9631;
0.9478	0.9466	0.9472	0.9516	0.9516;
0.9242	0.9236	0.9242	0.9326	0.9326;
0.8773	0.8778	0.877	0.8993	0.8993;
0.7812	0.7801	0.781	0.8094	0.8094];

ssim_snr5 = [0.8365	0.9627	0.9631	0.9526	0.9584;
0.8625	0.9639	0.9645	0.956	0.9595;
0.9036	0.9641	0.9651	0.9588	0.961;
0.9297	0.9641	0.965	0.9601	0.9605;
0.9446	0.9612	0.9623	0.9604	0.9597;
0.9508	0.9558	0.9564	0.9576	0.9572;
0.9442	0.9452	0.9454	0.9493	0.9492;
0.9232	0.9236	0.9236	0.932	0.9319;
0.8772	0.8772	0.8769	0.8992	0.8992;
0.7812	0.7805	0.781	0.8094	0.8094];

ssim_snr3 = [0.7263	0.9523	0.9522	0.9299	0.942;
0.7705	0.9544	0.9546	0.9342	0.944;
0.8176	0.955	0.9563	0.9409	0.946;
0.8638	0.9556	0.9574	0.9452	0.9473;
0.9019	0.9564	0.9564	0.9476	0.9482;
0.9278	0.9517	0.9523	0.9478	0.9478;
0.9352	0.9423	0.9429	0.9444	0.9441;
0.9209	0.9221	0.9228	0.9305	0.9304;
0.8769	0.8774	0.8769	0.899	0.899;
0.7812	0.7799	0.7805	0.8094	0.8094];

ssim_snr2 = [0.6392	0.9421	0.9403	0.9116	0.9258;
0.6816	0.9448	0.9444	0.917	0.9282;
0.7314	0.9471	0.9474	0.9223	0.9307;
0.7873	0.9488	0.9496	0.9274	0.9334;
0.8427	0.9491	0.9498	0.9316	0.937;
0.8904	0.9468	0.9473	0.9340	0.936;
0.9183	0.9393	0.9399	0.9336	0.9352;
0.9163	0.9207	0.9206	0.9276	0.9276;
0.8763	0.8771	0.8768	0.8985	0.8985;
0.7798	0.7805	0.7812	0.8094	0.8094];

mse_snr10 = [0.0531	0.0528	0.0514	0.0442	0.0416;
0.0478	0.0521	0.0458	0.0438	0.0434;
0.0462	0.0507	0.0476	0.0475	0.0471;
0.0481	0.0546	0.0524	0.0543	0.0542;
0.0561	0.0628	0.0601	0.0651	0.0650;
0.0674	0.0737	0.0709	0.079	0.079;
0.0832	0.0886	0.0862	0.098	0.098;
0.1085	0.1123	0.1108	0.1226	0.1226;
0.1569	0.1592	0.159	0.1575	0.1575;
0.2725	0.2736	0.2726	0.2637	0.2637];

mse_snr5 = [0.1073	0.0617	0.0622	0.0738	0.0658;
0.0891	0.0601	0.057	0.0689	0.0647;
0.0751	0.0592	0.058	0.066	0.0649;
0.0661	0.0582	0.0586	0.066	0.066;
0.0652	0.0651	0.0647	0.0709	0.0708;
0.0713	0.0752	0.0722	0.081	0.0808;
0.0845	0.0884	0.0866	0.0983	0.0981;
0.1089	0.1124	0.1103	0.1225	0.1225;
0.157	0.1592	0.1579	0.1576	0.1576;
0.2725	0.2735	0.2728	0.2637	0.2637];

mse_snr3 = [0.1799	0.0691	0.0678	0.1036	0.0857;
0.1469	0.0678	0.0665	0.0951	0.0831;
0.1188	0.0673	0.0657	0.0876	0.0811;
0.0964	0.0663	0.0644	0.0823	0.0808;
0.0831	0.0662	0.066	0.0812	0.0811;
0.0798	0.0767	0.075	0.086	0.0858;
0.0876	0.0901	0.0878	0.0998	0.099;
0.1097	0.113	0.111	0.1227	0.1227;
0.1571	0.1596	0.1583	0.1576	0.1576;
0.2725	0.2739	0.2731	0.2638	0.2638];

mse_snr2 = [0.2699	0.0793	0.0763	0.1196	0.0978;
0.2196	0.0768	0.0752	0.1105	0.095;
0.1754	0.0749	0.0742	0.102	0.0924;
0.138	0.0746	0.0724	0.0951	0.091;
0.1102	0.0742	0.0736	0.0933	0.0919;
0.0943	0.0803	0.0789	0.0924	0.0922;
0.0934	0.0917	0.0899	0.1025	0.1025;
0.1112	0.1136	0.1119	0.1233	0.1233;
0.1573	0.1594	0.1582	0.1578	0.1578;
0.2725	0.2819	0.2736	0.2638	0.2638];

%% UNet 1-NRMSE
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(1-mse_snr10(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr10(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr10(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(3, 1-mse_snr10(3,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr10(2,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
ylabel('1-NRMSE', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(1-mse_snr5(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr5(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr5(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(4, 1-mse_snr5(4,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr5(2,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)');
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(1-mse_snr3(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr3(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr3(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr3(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr3(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(1-mse_snr2(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr2(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr2(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr2(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr2(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2'); 
pbaspect([1 1 1]);
xlim([1,8]);

sgtitle('UNet Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold')
set(gcf,'Position',[100 100 2000 500])

%% UNet SSIM
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(ssim_snr10(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr10(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr10(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(3, ssim_snr10(3,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, ssim_snr10(1,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
ylabel('SSIM', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(ssim_snr5(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr5(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr5(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(4, ssim_snr5(4,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(3, ssim_snr5(3,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)');
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(ssim_snr3(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr3(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr3(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr3(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, ssim_snr3(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(ssim_snr2(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr2(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr2(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr2(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr2(5,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2'); 
pbaspect([1 1 1]);
xlim([1,8]);

sgtitle('UNet Performances (SSIM) at Different SNRs', 'FontWeight', 'bold')
set(gcf,'Position',[100 100 2000 500])

%% TV 1-NRMSE
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(1-mse_snr10(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr10(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr10(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(2, 1-mse_snr10(2,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, 1-mse_snr10(1,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
ylabel('1-NRMSE', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(1-mse_snr5(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr5(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr5(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(4, 1-mse_snr5(4,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr5(2,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)');
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(1-mse_snr3(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr3(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr3(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr3(5,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr3(4,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(1-mse_snr2(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr2(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr2(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, 1-mse_snr2(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr2(4,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2'); 
pbaspect([1 1 1]);
xlim([1,8]);

sgtitle('SENSE-TV Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold')
set(gcf,'Position',[100 100 2000 500])


%% TV SSIM
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(ssim_snr10(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr10(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr10(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(2, ssim_snr10(2,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, ssim_snr10(1,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
ylabel('SSIM', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(ssim_snr5(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr5(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr5(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr5(5,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(3, ssim_snr5(3,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)');
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(ssim_snr3(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr3(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr3(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, ssim_snr3(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr3(5,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3'); 
pbaspect([1 1 1]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(ssim_snr2(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr2(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr2(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, ssim_snr2(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr2(5,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)');
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2'); 
pbaspect([1 1 1]);
xlim([1,8]);

sgtitle('SENSE-TV Performances (SSIM) at Different SNRs', 'FontWeight', 'bold')
set(gcf,'Position',[100 100 2000 500])