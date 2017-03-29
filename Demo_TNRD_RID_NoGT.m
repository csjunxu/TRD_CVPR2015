
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

noise_levels     =  [10];
images           =  dir(fullfile(ref_folder,'*.png'));
format compact;

for i = 1 : numel(images)
    [~, name, exte]  =  fileparts(images(i).name);
    I =   double(imread( fullfile(ref_folder,images(i).name) ));
    [R,C,ch] = size(I);
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
        IMout = zeros(size(I));
        for c = 1:ch
            %% denoising
            input = pad(I(:,:,c));
            noisy = pad(I(:,:,c));
            for s = 1:stage
                deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
                t = crop(deImg);
                deImg = pad(t);
                input = deImg;
            end
            x_star = max(0, min(t(:), 255));
            IMoutcc = reshape(x_star,R,C);
            IMout(:,:,c) = IMoutcc;
        end
        imwrite(IMout/255, ['C:/Users/csjunxu/Desktop/ICCV2017/1nc_Results/Real_' method '/' method '_Real_' num2str(noise_levels) '_' name '.png']);
        
    end
end



