% =========================================================
% Xianjing Meng Mail: xianjingmeng@foxmail.com
% ͼ��Ԥ����ѡȡͼ�����Ȥ����Ĳ��֣�����һ���ļ���
% ͼ������ img_a_b.bmp a:�ڼ�����ָ b:�ڼ���ͼ��
% 
% ��һ���⣨0����30���ˣ�û��6����ָ��ÿ����ָͼ����Ŀ����
% ע��һ��matlab��̹淶 0-�� 255-��
% =========================================================

addpath(genpath(pwd));
        t=6;
        img=imread('E:\s1.bmp');
%         img=mla_nomalization(img,11);%1.�Ҷȹ�һ�� 
%         img = image_pre_process(img) % mat
%           img=histeq(img,256); % pang
%         img = MMSDG(img); %3 peng
%         img = ICGF(img); %4 kim
        
        img=uint8(img(t+1:end-t,t+1:end-t));
        imshow(img);
        imwrite(imresize(img,1,'nearest'),'E:\yuchuli\s7.bmp');
