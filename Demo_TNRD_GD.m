
%--------------------------------------------------------------------------
clc;
clear;
%%% addpath
addpath('..\RandNoise')
load RandNoise.mat;

addpath(genpath('./.'))
method           =  '12';
ref_folder       =  '..\Ref_Gray';
den_folder       =  method;

if ~isdir(den_folder)
    mkdir(den_folder)
end

noise_levels     =  [10, 25, 35, 50, 75];
images           =  dir(fullfile(ref_folder,'*.bmp'));
format compact;



for i = 1 : numel(images)
    
    [~, name, exte]  =  fileparts(images(i).name);
    I =   double(imread( fullfile(ref_folder,images(i).name) ));
    [R,C] = size(I);
    for j = 1 : numel(noise_levels)
        disp([i,j]);
        nSig             =    noise_levels(j);
        
        noise_img          =   I+ nSig*RandNoise{i,j};
        
        if nSig == 10
            load JointTraining_7x7_400_180x180_stage=5_sigma=10.mat;
            10
        elseif nSig == 25
            load JointTraining_7x7_400_180x180_stage=5_sigma=25.mat;
            25
        elseif nSig == 35
            load JointTraining_7x7_400_180x180_stage=5_sigma=35.mat;
            35
        elseif nSig == 50
            load JointTraining_7x7_400_180x180_stage=5_sigma=50.mat;
            50
        elseif nSig == 75
            load JointTraining_7x7_400_180x180_stage=5_sigma=75.mat;
            75
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
        imwrite(im/255, fullfile(den_folder, [name, num2str(j), method,exte] ));
        
        
    end
end




