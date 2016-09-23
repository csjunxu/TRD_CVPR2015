
%--------------------------------------------------------------------------
clc;
clear;

setname          = 'Real_NoisyImage';
method           =  'TNRD';
ref_folder       =  fullfile('C:\Users\csjunxu\Desktop\CVPR2017\1_Results\',setname);

den_folder       =  ['Results_',setname,'_',method];
if ~isdir(den_folder)
    mkdir(den_folder)
end

noise_levels     =  [15];
images           =  dir(fullfile(ref_folder,'*.png'));
format compact;

for i = 1 : numel(images)
    [~, name, exte]  =  fileparts(images(i).name);
    I =   im2double(imread( fullfile(ref_folder,images(i).name) ));
    [R,C,ch] = size(I);
                if ch==1
                IMin_y = I;
            else
                % change color space, work on illuminance only
                IMin_ycbcr = rgb2ycbcr(I);
                IMin_y = IMin_ycbcr(:, :, 1);
                IMin_cb = IMin_ycbcr(:, :, 2);
                IMin_cr = IMin_ycbcr(:, :, 3);
            end
    for j = 1 : numel(noise_levels)
        disp([i,j]);
        nSig               =    noise_levels(j);
%         randn('seed',0);
%         IMin_y          =   I+ nSig*randn(size(I));
       % IMin_y = double(uint8(IMin_y));
        
        if nSig == 10
            load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
            10
        elseif nSig == 15
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
        input = pad(IMin_y);
        noisy = pad(IMin_y);
        for s = 1:stage
            deImg = denoisingOneStepGMixMFs(noisy*255, input, trained_model{s});
            t = crop(deImg);
            deImg = pad(t);
            input = deImg;
        end
        x_star = max(0, min(t(:), 255));
        IMout_y = reshape(x_star,R,C);
%         [psnr_cur, ssim_cur] = Cal_PSNRSSIM(I,im,0,0);
%         imshow(im/255);drawnow;
      %  PSNR_value = csnr(uint8(I),uint8(im),0,0);
                  if ch==1
                IMout = IMout_y/255;
            else
                IMout_ycbcr = zeros(size(I));
                IMout_ycbcr(:, :, 1) = IMout_y/255;
                IMout_ycbcr(:, :, 2) = IMin_cb;
                IMout_ycbcr(:, :, 3) = IMin_cr;
                IMout = ycbcr2rgb(IMout_ycbcr);
            end
         imwrite(IMout, ['C:\Users\csjunxu\Desktop\CVPR2017\1_Results\Real_' method '\' method '_Real_' num2str(noise_levels) '_' name '.png']);
 %         PSNR(i,j) = psnr_cur;
%         SSIM(i,j) = ssim_cur;
    end
end

% mean_PSNR_value = mean(PSNR,1);
% mean_SSIM_value = mean(SSIM,1);

%save(['PSNR_',setname,'_',method],'noise_levels','PSNR','mean_value');


