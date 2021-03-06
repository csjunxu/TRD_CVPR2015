
clear;
Original_image_dir  =    'C:\Users\csjunxu\Desktop\TWSCGIN\cleanimages\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

method = 'TNRD';
writematpath = 'C:/Users/csjunxu/Desktop/ECCV2018 Denoising/Results_AWGN/';
writefilepath  = [writematpath method '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end

for nSig     =  [15 25 35 50 75]
    
    PSNR = [];
    SSIM = [];
    
    for i = 1 : im_num
        
        I =   double(imread( fullfile(Original_image_dir, im_dir(i).name) ));
        [R,C] = size(I);
        randn('seed',0);
        nI          =   I+ nSig*randn(size(I));
        
        if nSig == 15
            load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
        elseif nSig == 25
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
        elseif nSig == 35
            load JointTraining_7x7_400_180x180_stage=5_sigma=35.mat;
        elseif nSig == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
        elseif nSig == 75
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
        elseif nSig >75 
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
        end
        nSig
        %% default setting
        filter_size = 7;
        m = filter_size^2 - 1;
        filter_num = 48;
        BASIS = gen_dct2(filter_size);
        BASIS = BASIS(:,2:end);
        %% pad and crop operation
        bsz = filter_size+1;
        bndry = [bsz,bsz];
        pad   = @(x) padarray(x,bndry,'symmetric','both');
        crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
        %% MFs means and precisions
        KernelPara.fsz = filter_size;
        KernelPara.filtN = filter_num;
        KernelPara.basis = BASIS;
        trained_model = save_trained_model(cof, MFS, stage, KernelPara);
        
        input = pad(nI);
        noisy = pad(nI);
        for s = 1:stage
            deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
            t = crop(deImg);
            deImg = pad(t);
            input = deImg;
        end
        x_star = max(0, min(t(:), 255));
        im = reshape(x_star,R,C);
        % im = I + (im - I)*100/60;
        imname = sprintf([writefilepath method '_nSig' num2str(nSig)  '_' im_dir(i).name]);
        imwrite(im/255, imname);
        PSNR = [PSNR  csnr( im, I, 0, 0 )];
        SSIM  = [SSIM cal_ssim( im, I, 0, 0 )];
        fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n', im_dir(i).name, PSNR(end), SSIM(end)  );
    end
    mPSNR=mean(PSNR);
    mSSIM=mean(SSIM);
    fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR,mSSIM);
    name = sprintf([writematpath method '_nSig' num2str(nSig) '.mat']);
    save(name, 'nSig','PSNR','SSIM','mPSNR','mSSIM');
end




