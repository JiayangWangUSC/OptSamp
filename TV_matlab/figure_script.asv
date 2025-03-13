%% figure 
clear all;
close all;
clc;
N1 = 320; N2 = 320; Nc = 20; Ns =8;

%% plot NRMSE-resolution trend
ssim_snr10 = [0.9409	0.9719	0.9757	0.9752	0.9786;
0.9542	0.9725	0.9757	0.9755	0.9784;
0.9638	0.9729	0.9746	0.9755	0.9775;
0.969	0.9709	0.9726	0.9742	0.9758;
0.9671	0.9665	0.9681	0.9701	0.9718;
0.9608	0.9594	0.9599	0.9632	0.9651;
0.9478	0.9466	0.9472	0.9516	0.9536;
0.9242	0.9236	0.9236	0.9326	0.9326;
0.8773	0.8778	0.8778	0.8993	0.8993;
0.7812	0.7801	0.7801	0.8094	0.8094];

ssim_snr5 = [0.8365	0.9627	0.9641	0.9526	0.9604;
0.8625	0.9639	0.9655	0.956	0.9615;
0.9036	0.9641	0.9661	0.9588	0.963;
0.9297	0.9641	0.966	0.9601	0.9627;
0.9446	0.9612	0.9633	0.9604	0.9625;
0.9508	0.9558	0.9564	0.9576	0.9582;
0.9442	0.9452	0.9454	0.9493	0.9502;
0.9232	0.9236	0.9236	0.932	0.9329;
0.8772	0.8772	0.8772	0.8992	0.9002;
0.7812	0.7805	0.7805	0.8094	0.8104];

ssim_snr3 = [0.7263	0.9523	0.9532	0.9299	0.944;
0.7705	0.9544	0.9556	0.9342	0.947;
0.8176	0.955	0.9573	0.9409	0.948;
0.8638	0.9556	0.9584	0.9452	0.9493;
0.9019	0.9564	0.9574	0.9476	0.9502;
0.9278	0.9517	0.9523	0.9478	0.9498;
0.9352	0.9423	0.9429	0.9444	0.9461;
0.9209	0.9221	0.9221	0.9305	0.9314;
0.8769	0.8774	0.8774	0.899	0.9000;
0.7812	0.7799	0.7799	0.8094	0.8104];

ssim_snr2 = [0.6392	0.9421	0.9413	0.9116	0.9278;
0.6816	0.9448	0.9454	0.917	0.9302;
0.7314	0.9471	0.9484	0.9223	0.9327;
0.7873	0.9488	0.9506	0.9274	0.9354;
0.8427	0.9491	0.9508	0.9316	0.9379;
0.8904	0.9468	0.9473	0.9340	0.9365;
0.9183	0.9393	0.9399	0.9336	0.9347;
0.9163	0.9207	0.9207	0.9276	0.9286;
0.8763	0.8771	0.8771	0.8985	0.8995;
0.7798	0.7805	0.7805	0.8094	0.8094];

mse_snr10 = [0.0531	0.0528	0.0509	0.0442	0.0416;
0.0478	0.0521	0.0448	0.0438	0.0424;
0.0462	0.0507	0.0466	0.0475	0.0461;
0.0481	0.0546	0.0514	0.0543	0.0532;
0.0561	0.0628	0.0591	0.0651	0.0640;
0.0674	0.0737	0.0709	0.079	0.078;
0.0832	0.0886	0.0862	0.098	0.097;
0.1085	0.1123	0.1123	0.1226	0.1226;
0.1569	0.1592	0.1592	0.1575	0.1575;
0.2725	0.2736	0.2736	0.2637	0.2637];

mse_snr5 = [0.1073	0.0617	0.0612	0.0738	0.0647;
0.0891	0.0601	0.056	0.0689	0.0636;
0.0751	0.0592	0.057	0.066	0.0638;
0.0661	0.0582	0.0576	0.066	0.0649;
0.0652	0.0651	0.0637	0.0709	0.0697;
0.0713	0.0752	0.0722	0.081	0.079;
0.0845	0.0884	0.0866	0.0983	0.0971;
0.1089	0.1124	0.1124	0.1225	0.1225;
0.157	0.1592	0.1592	0.1576	0.1576;
0.2725	0.2735	0.2735	0.2637	0.2637];

mse_snr3 = [0.1799	0.0691	0.0658	0.1036	0.0837;
0.1469	0.0678	0.0655	0.0951	0.0811;
0.1188	0.0673	0.0647	0.0876	0.0791;
0.0964	0.0663	0.0634	0.0823	0.0788;
0.0831	0.0662	0.065	0.0812	0.0791;
0.0798	0.0767	0.076	0.086	0.0858;
0.0876	0.0901	0.0888	0.0998	0.099;
0.1097	0.113	0.113	0.1227	0.1227;
0.1571	0.1596	0.1596	0.1576	0.1576;
0.2725	0.2739	0.2739	0.2638	0.2638];

mse_snr2 = [0.2699	0.0793	0.0753	0.1196	0.0958;
0.2196	0.0768	0.0742	0.1105	0.093;
0.1754	0.0749	0.0732	0.102	0.0904;
0.138	0.0746	0.0714	0.0951	0.089;
0.1102	0.0742	0.0726	0.0933	0.0899;
0.0943	0.0803	0.0799	0.0924	0.0912;
0.0934	0.0917	0.0909	0.1025	0.1015;
0.1112	0.1136	0.1136	0.1233	0.1233;
0.1573	0.1594	0.1594	0.1578	0.1578;
0.2725	0.2819	0.2819	0.2638	0.2638];

fft_ssim_snr10 = [0.9277	0.9267;
0.9424	0.9421;
0.9499	0.9509;
0.9495	0.9524;
0.9395	0.9441;
0.9232	0.9288;
0.9018	0.906;
0.874	0.8716];

fft_ssim_snr5 = [0.7877	0.7992;
0.8327	0.8374;
0.8737	0.8738;
0.9068	0.9046;
0.9261	0.9249;
0.93	0.9311;
0.9159	0.9185;
0.8819	0.8825];

fft_ssim_snr3 = [0.6608	0.7208;
0.7076	0.7484;
0.7595	0.7828;
0.8127	0.8219;
0.8615	0.8622;
0.8984	0.8961;
0.9125	0.9114;
0.8913	0.8913];

fft_ssim_snr2 = [0.6048	0.7441;
0.634	0.7397;
0.6736	0.7464;
0.6988	0.7654;
0.7786	0.7985;
0.8359	0.8401;
0.8809	0.8789;
0.8889	0.8882];

fft_mse_snr10 = [0.0643	0.0594;
0.0636	0.0579;
0.063	0.0584;
0.0659	0.0608;
0.0718	0.0672;
0.08	0.0762;
0.0921	0.0896;
0.1136	0.113];

fft_mse_snr5 = [0.097	0.0917;
0.0875	0.0835;
0.0804	0.0771;
0.0763	0.0731;
0.0751	0.0739;
0.0822	0.0792;
0.0928	0.0905;
0.1137	0.1135];

fft_mse_snr3 = [0.1364	0.1185;
0.1223	0.1106;
0.1089	0.102;
0.0975	0.0937;
0.0906	0.0883;
0.0892	0.0873;
0.0954	0.0938;
0.1143	0.1138];

fft_mse_snr2 = [0.158	0.1236;
0.152	0.1226;
0.1381	0.1192;
0.1186	0.1134;
0.1109	0.1062;
0.1022	0.1004;
0.1016	0.1007;
0.116	0.1158];

%% fft 1-NRMSE
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(1-fft_mse_snr10(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-fft_mse_snr10(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-fft_mse_snr10(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(2, 1-fft_mse_snr10(2,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-fft_mse_snr10(2,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
ylabel('1-NRMSE', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(1-fft_mse_snr5(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-fft_mse_snr5(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-fft_mse_snr5(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(4, 1-fft_mse_snr5(4,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-fft_mse_snr5(4,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)', 'FontSize', 12);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(1-fft_mse_snr3(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-fft_mse_snr3(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-fft_mse_snr3(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(6, 1-fft_mse_snr3(6,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(6, 1-fft_mse_snr3(6,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(1-fft_mse_snr2(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-fft_mse_snr2(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-fft_mse_snr2(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(7, 1-fft_mse_snr2(7,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(6, 1-fft_mse_snr2(6,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);


set(gcf,'Position',[100 100 2000 500])
sgtitle('Apodization Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);


% Save as PDF and EPS
print(gcf, '-dpdf', 'fig5.pdf');
print(gcf, '-depsc', 'fig5.eps');

%% fft SSIM
figure();

% SNR 10
subplot(1,4,1); 
plot(1:8,(fft_ssim_snr10(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,fft_ssim_snr10(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,fft_ssim_snr10(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(4, fft_ssim_snr10(4,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, fft_ssim_snr10(4,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
ylabel('SSIM', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.6,0.96]);
xlim([1,8]);

% SNR 5
subplot(1,4,2);
plot(1:8,(fft_ssim_snr5(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,fft_ssim_snr5(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,fft_ssim_snr5(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(6, fft_ssim_snr5(6,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(6, fft_ssim_snr5(6,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)', 'FontSize', 12);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.6,0.96]);
xlim([1,8]);

% SNR 3
subplot(1,4,3);
plot(1:8,(fft_ssim_snr3(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,fft_ssim_snr3(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,fft_ssim_snr3(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(7, fft_ssim_snr3(7,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(7, fft_ssim_snr3(7,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.6,0.96]);
xlim([1,8]);

% SNR 2
subplot(1,4,4);
plot(1:8,(fft_ssim_snr2(1,1))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,fft_ssim_snr2(1:8,1),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,fft_ssim_snr2(1:8,2),'Color',[0.3,0.7,0.9]);hold on;
plot(8, fft_ssim_snr2(8,1), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(8, fft_ssim_snr2(8,2), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.6,0.96]);
xlim([1,8]);

set(gcf,'Position',[100 100 2000 500])
sgtitle('Apodization Performances (SSIM) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);

% Save as PDF and EPS
print(gcf, '-dpdf', 'fig6.pdf');
print(gcf, '-depsc', 'fig6.eps');

%% UNet 1-NRMSE
figure();

% SNR 10
ax1 = subplot(2,4,1); 
plot(1:8,(1-mse_snr10(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr10(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr10(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(3, 1-mse_snr10(3,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr10(2,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
ylabel('1-NRMSE', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10','FontSize', 14); 
%set(gca, 'XTickLabel', []);
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 5
ax2 = subplot(2,4,2);
plot(1:8,(1-mse_snr5(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr5(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr5(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(4, 1-mse_snr5(4,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr5(2,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5','FontSize', 14); 
%set(gca, 'XTickLabel', []);
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 3
ax3 = subplot(2,4,3);
plot(1:8,(1-mse_snr3(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr3(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr3(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr3(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr3(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%set(gca, 'XTickLabel', []);
title('SNR = 3','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 2
ax4 = subplot(2,4,4);
plot(1:8,(1-mse_snr2(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr2(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr2(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr2(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr2(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%set(gca, 'XTickLabel', []);
title('SNR = 2','FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

%sgtitle('UNet Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold')
%set(gcf,'Position',[100 100 2000 500])


%set(gcf, 'DefaultAxesFontSize', 18);
%set(gcf, 'DefaultTextFontSize', 18);
%ylabel('1-NRMSE', 'FontSize', 18); % Example for ylabel
%sgtitle('UNet Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);

%... (Rest of your plotting code)...

% Save as PDF and EPS
%print(gcf, '-dpdf', 'fig1.pdf');
%print(gcf, '-depsc', 'fig1.eps');

% UNet SSIM
%figure();

% SNR 10
ax5 = subplot(2,4,5); 
plot(1:8,(ssim_snr10(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr10(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr10(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(3, ssim_snr10(3,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, ssim_snr10(1,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
ylabel('SSIM', 'FontSize', 14);
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 10', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 5
ax6 = subplot(2,4,6);
plot(1:8,(ssim_snr5(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr5(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr5(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(4, ssim_snr5(4,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(3, ssim_snr5(3,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 5', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 3
ax7 = subplot(2,4,7);
plot(1:8,(ssim_snr3(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr3(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr3(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr3(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, ssim_snr3(4,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 3', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 2
ax8 = subplot(2,4,8);
plot(1:8,(ssim_snr2(1,2))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr2(1:8,2),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr2(1:8,3),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr2(5,2), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr2(5,3), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 2', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);


% Get positions
pos5 = get(ax5, 'Position');
pos6 = get(ax6, 'Position');
pos7 = get(ax7, 'Position');
pos8 = get(ax8, 'Position');

% Adjust the bottom position to reduce the gap
adjustment = - 0.12; % Adjust this value as needed

pos5(2) = pos5(2) - adjustment;
pos6(2) = pos6(2) - adjustment;
pos7(2) = pos7(2) - adjustment;
pos8(2) = pos8(2) - adjustment;

% Set the new positions
set(ax5, 'Position', pos5);
set(ax6, 'Position', pos6);
set(ax7, 'Position', pos7);
set(ax8, 'Position', pos8);

set(gcf,'Position',[100 100 2000 1000])
%sgtitle('UNet Performances (SSIM) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);

print(gcf, '-dpdf', 'plot_unet.pdf');
print(gcf, '-depsc', 'plot_unet.eps');

%% TV 1-NRMSE
figure();

% SNR 10
ax1 = subplot(2,4,1); 
plot(1:8,(1-mse_snr10(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr10(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr10(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(2, 1-mse_snr10(2,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, 1-mse_snr10(1,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
ylabel('1-NRMSE', 'FontSize', 14);
xticks(1:8); 
legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 10', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 5
ax2 = subplot(2,4,2);
plot(1:8,(1-mse_snr5(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr5(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr5(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(4, 1-mse_snr5(4,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(2, 1-mse_snr5(2,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 5', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 3
ax3 = subplot(2,4,3);
plot(1:8,(1-mse_snr3(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr3(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr3(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(5, 1-mse_snr3(5,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr3(4,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 3', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% SNR 2
ax4 = subplot(2,4,4);
plot(1:8,(1-mse_snr2(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,1-mse_snr2(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,1-mse_snr2(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, 1-mse_snr2(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(4, 1-mse_snr2(4,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
%xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
title('SNR = 2', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.84,0.96]);
xlim([1,8]);

% set(gcf,'Position',[100 100 2000 500])
% sgtitle('SENSE-TV Performances (1-NRMSE) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);
% 
% print(gcf, '-dpdf', 'fig3.pdf');
% print(gcf, '-depsc', 'fig3.eps');

% TV SSIM
%figure();

% SNR 10
ax5 = subplot(2,4,5); 
plot(1:8,(ssim_snr10(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr10(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr10(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(2, ssim_snr10(2,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(1, ssim_snr10(1,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
ylabel('SSIM', 'FontSize', 14);
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 10', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 5
ax6 = subplot(2,4,6);
plot(1:8,(ssim_snr5(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr5(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr5(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(5, ssim_snr5(5,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(3, ssim_snr5(3,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6); 
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE'); % Only need ylabel on the leftmost plot
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 5', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 3
ax7 = subplot(2,4,7);
plot(1:8,(ssim_snr3(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr3(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr3(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, ssim_snr3(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr3(5,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 3', 'FontSize', 14); 
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);

% SNR 2
ax8 = subplot(2,4,8);
plot(1:8,(ssim_snr2(1,4))*ones(1,8),'--','Color',[0.5,0.5,0.5]);hold on;
plot(1:8,ssim_snr2(1:8,4),'Color',[0.3,0.5,0.9]);hold on;
plot(1:8,ssim_snr2(1:8,5),'Color',[0.3,0.7,0.9]);hold on;
plot(6, ssim_snr2(6,4), 'o','Color',[0.3,0.5,0.9], 'MarkerSize', 6); 
plot(5, ssim_snr2(5,5), 'o', 'Color',[0.3,0.7,0.9], 'MarkerSize', 6);  
xlabel('Resolution(%)', 'FontSize', 12);
%ylabel('1-NRMSE');
xticks(1:8); 
%legend('Baseline','Uniform','Optimized','Location','southwest')
xticklabels({'100', '90', '80', '70', '60','50','40','30'});
%title('SNR = 2', 'FontSize', 14);
pbaspect([1 1 1]);
ylim([0.9,0.98]);
xlim([1,8]);


% Get positions
pos5 = get(ax5, 'Position');
pos6 = get(ax6, 'Position');
pos7 = get(ax7, 'Position');
pos8 = get(ax8, 'Position');

% Adjust the bottom position to reduce the gap
adjustment = - 0.12; % Adjust this value as needed

pos5(2) = pos5(2) - adjustment;
pos6(2) = pos6(2) - adjustment;
pos7(2) = pos7(2) - adjustment;
pos8(2) = pos8(2) - adjustment;

% Set the new positions
set(ax5, 'Position', pos5);
set(ax6, 'Position', pos6);
set(ax7, 'Position', pos7);
set(ax8, 'Position', pos8);


set(gcf,'Position',[100 100 2000 1000])
%sgtitle('SENSE-TV Performances (SSIM) at Different SNRs', 'FontWeight', 'bold', 'FontSize', 16);

print(gcf, '-dpdf', 'plot_tv.pdf');
print(gcf, '-depsc', 'plot_tv.eps');





