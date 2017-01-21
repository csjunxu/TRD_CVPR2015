
%--------------------------------------------------------------------------
clc;
clear;

setname          = 'Set68';
method           =  'TNRD';
ref_folder       =  fullfile('D:\matlab1\DL_code\ECCV_codes\ECCV_test\Test',setname);

den_folder       =  ['Results_',setname,'_',method];
if ~isdir(den_folder)
    mkdir(den_folder)
end

noise_levels     =  [20];
images           =  dir(fullfile(ref_folder,'*.png'));
format compact;

for i = 1 : numel(images)
    [~, name, exte]  =  fileparts(images(i).name);
    I =   double(imread( fullfile(ref_folder,images(i).name) ));
    [R,C] = size(I);
    
   
    for j = 1 : numel(noise_levels)
        disp([i,j]);
        nSig               =    noise_levels(j);
        randn('seed',0);
        
        
        noise_img          =   I+ nSig*randn(size(I));
       % noise_img = double(uint8(noise_img));
        
        if nSig == 15
            load JointTraining_7x7_400_180x180_stage=5_sigma=15.mat;
            15
        elseif nSig == 20
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
            25
        elseif nSig == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
            50
        end
        
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
        
        
        input = pad(noise_img);
        noisy = pad(noise_img);
        for s = 1:stage
            deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
            t = crop(deImg);
            deImg = pad(t);
            input = deImg;
        end
        x_star = max(0, min(t(:), 255));
        im = reshape(x_star,R,C);
        
        [psnr_cur, ssim_cur] = Cal_PSNRSSIM(I,im,0,0);
        imshow(im/255);drawnow;
      %  PSNR_value = csnr(uint8(I),uint8(im),0,0);
     %   imwrite(im/255, fullfile(den_folder, [name, '_sigma=' num2str(nSig,'%02d'),'_',method,'_PSNR=',num2str(PSNR_value,'%2.2f'), exte] ));
        PSNR(i,j) = psnr_cur;
        SSIM(i,j) = ssim_cur;
    end
end

mean_PSNR_value = mean(PSNR,1);
mean_SSIM_value = mean(SSIM,1);

%save(['PSNR_',setname,'_',method],'noise_levels','PSNR','mean_value');


