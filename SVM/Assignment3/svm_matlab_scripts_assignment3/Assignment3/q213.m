close all;

load digits; 
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

% Add noise to the digit maps
noisefactor = 1.0;
noise = noisefactor*maxx; % sd for Gaussian noise

filename_train = 'Data/reconstruction_errors_Xtrain.csv';
header = {'nc','sig2', 'MSE'};
writecell(header, filename_train);

filename_1 = 'Data/reconstruction_errors_Xtest1.csv';
header = {'nc','sig2', 'MSE'};
writecell(header, filename_1);

filename_2 = 'Data/reconstruction_errors_Xtest2.csv';
header = {'nc','sig2', 'MSE'};
writecell(header, filename_2);

Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt1 = Xtest1; 
for i=1:size(Xtest1,1);
  randn('state', N+i);
  Xnt1(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

Xnt2 = Xtest2; 
for i=1:size(Xtest2,1);
  randn('state', N+10+i);
  Xnt2(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end

% select training set
Xtr = Xn(1:1:end,:);

% rule of thumb
sig2 =dim*mean(var(Xtr)); 
sig2list = [log(1.1), log(1.5), log(2), log(5), log(10), log(50), log(100)].*sig2;
%sig2list = [0.001, 0.01, 0.1, 1, 10, 10, 1000];

npcs = [2.^(4:7) 190];
lpcs = length(npcs);


% on training data
ConErr = [];

for i = 1:length(sig2list),
    sig2 = sig2list(i);
    
    [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
    [lam, ids]=sort(-lam); 
    lam = -lam; 
    U=U(:,ids);

    for j = 1:lpcs,
        errlist = [];
        nc = npcs(j); 
        Ud=U(:,(1:nc)); 
        lamd=lam(1:nc);
        
        for k = [1:N],
            xt = Xn(k,:);
            Xdt(k,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
            err = sum((Xtr(k,:) - Xdt(k,:)).^2);
            errlist = [errlist; err];
        end
        conerr = sum(errlist);
        ConErr(i,j) = conerr;
        fprintf( 'The reconstruction error of nc = %d and sig2 = %f is %f\n', nc, sig2, conerr);
        % Save to file
        header = {nc, sig2, conerr};
        writecell(header, filename_train, 'WriteMode', 'append');
    end
end
        


% on test1 data set
ConErr1 = [];

[v1, h1] = size(Xtest1)
for i = 1:length(sig2list),
    sig2 = sig2list(i);
    
    [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
    [lam, ids]=sort(-lam); 
    lam = -lam; 
    U=U(:,ids);

    for j = 1:lpcs,
        errlist1 = [];
        nc = npcs(j); 
        Ud=U(:,(1:nc)); 
        lamd=lam(1:nc);
        
        for k = [1:v1],
            xt1 = Xnt1(k,:);
            Xdt1(k,:) = preimage_rbf(Xtr,sig2,Ud,xt1,'denoise');
            err1 = sum((Xtest1(k,:) - Xdt1(k,:)).^2);
            errlist1 = [errlist1; err1];
        end
        conerr1 = sum(errlist1);
        ConErr1(i,j) = conerr1;
        fprintf( 'The reconstruction error of nc = %d and sig2 = %f is %f\n', nc, sig2, conerr1);
        % Save to file
        header = {nc, sig2, conerr1};
        writecell(header, filename_1, 'WriteMode', 'append');
    end
end



% on test2 data set
ConErr2 = [];

[v2, h2] = size(Xtest2)
for i = 1:length(sig2list),
    sig2 = sig2list(i);
    
    [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
    [lam, ids]=sort(-lam); 
    lam = -lam; 
    U=U(:,ids);

    for j = 1:lpcs,
        errlist2 = [];
        nc = npcs(j); 
        Ud=U(:,(1:nc)); 
        lamd=lam(1:nc);
        
        for k = [1:v2],
            xt2 = Xnt2(k,:);
            Xdt2(k,:) = preimage_rbf(Xtr,sig2,Ud,xt2,'denoise');
            err2 = sum((Xtest2(k,:) - Xdt2(k,:)).^2);
            errlist2 = [errlist2; err2];
        end
        conerr2 = sum(errlist2);
        ConErr2(i,j) = conerr2;
        fprintf( 'The reconstruction error of nc = %d and sig2 = %f is %f\n', nc, sig2, conerr2);
        header = {nc, sig2, conerr2};
        writecell(header, filename_2, 'WriteMode', 'append');
    end
end


% Plots for hyperparameters analysis
for i = [1:7],
    figure;
    hold on;
    plot(npcs, ConErr(i,:)./N, '*-');
    plot(npcs, ConErr1(i,:)./v1, 'o-'); 
    plot(npcs, ConErr2(i,:)./v2, 'x-'); 
    xlabel('nc'), ylabel('reconstruction error');
    legend('Xtrain', 'Xtest1', 'Xtest2')
    title(['The reconstruction error of sig2 = ', num2str(sig2list(i))]);
    hold off;
end

disp("Saving results...");

writematrix(ConErr, "Data/conerr.txt");
writematrix(ConErr1, "Data/conerr.txt");


% After choosing the hyperparameters, we try again on two test data sets.
sig2 = log(1.5)*dim*mean(var(Xtr));
nc = 64;

[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); 
lam = -lam; 
U=U(:,ids);

Ud=U(:,(1:nc)); 
lamd=lam(1:nc);

for k = [1:v1],
    xt1now = Xnt1(k,:);
    Xdt1now(k,:) = preimage_rbf(Xtr,sig2,Ud,xt1now,'denoise');
end

for k = [1:v2],
    xt2now = Xnt2(k,:);
    Xdt2now(k,:) = preimage_rbf(Xtr,sig2,Ud,xt2now,'denoise');
end



% Plot of the results
figure; 
colormap(gray); 

for j = 1:10, 
    subplot(6, 10, j);
    pcolor(1:15,16:-1:1,reshape(Xtest1(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('original'),
    end 
end

for j = 1:10,
    subplot(6, 10, 10+j);
    pcolor(1:15,16:-1:1,reshape(Xnt1(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('noisy'),
    end 
end 

for j = 1:10,
    subplot(6, 10, 20+j);
    pcolor(1:15,16:-1:1,reshape(Xdt1now(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('denoised'),
    end 
end


for j = 1:10, 
    subplot(6, 10, 30+j);
    pcolor(1:15,16:-1:1,reshape(Xtest2(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('original'),
    end 
end

for j = 1:10,
    subplot(6, 10, 40+j);
    pcolor(1:15,16:-1:1,reshape(Xnt2(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('noisy'),
    end 
end 

for j = 1:10,
    subplot(6, 10, 50+j);
    pcolor(1:15,16:-1:1,reshape(Xdt2now(j,:), 15, 16)'); 
    shading interp; 
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if j == 1,
        ylabel('denoised'),
    end 
end