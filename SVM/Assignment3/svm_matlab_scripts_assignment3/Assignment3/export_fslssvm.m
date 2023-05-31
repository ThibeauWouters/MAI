
figure;
boxplot(e,'Label',user_process);
ylabel('Error estimate');
title('Errors');
exportgraphics(gcf,"Plots/fslssvm_err.png", 'Resolution', 600);


figure;
boxplot(s,'Label',user_process);
ylabel('SV estimate');
title('Number of SV');
exportgraphics(gcf,"Plots/fslssvm_sv.png", 'Resolution', 600);


figure;
boxplot(t,'Label',user_process);
ylabel('Time estimate');
title('Time taken');
exportgraphics(gcf,"Plots/fslssvm_time.png", 'Resolution', 600);