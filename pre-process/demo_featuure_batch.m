% =========================================================
% Xianjing Meng Mail: xianjingmeng@foxmail.com
% 图像预处理：选取图像感兴趣区域的部分，存入一个文件夹
% 图像名称 img_a_b.bmp a:第几个手指 b:第几个图像
% 
% 第一个库（0）：30个人，没人6根手指，每个手指图像数目不等
% 注意一下matlab编程规范 0-黑 255-白
% =========================================================

addpath(genpath(pwd));
        t=6;
        img=imread('E:\s1.bmp');
%         img=mla_nomalization(img,11);%1.灰度归一化 
%         img = image_pre_process(img) % mat
%           img=histeq(img,256); % pang
%         img = MMSDG(img); %3 peng
%         img = ICGF(img); %4 kim
        
        img=uint8(img(t+1:end-t,t+1:end-t));
        imshow(img);
        imwrite(imresize(img,1,'nearest'),'E:\yuchuli\s7.bmp');
