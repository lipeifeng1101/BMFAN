close all;clear all;clc
a=imread('C:\Users\HP\Desktop\File\tutor\summer\image\polyu database 3126ROI\1_1.bmp');
subplot(2,3,1);imshow(a);title('ԭͼ');
subplot(2,3,4);imhist(a);title('ԭͼֱ��ͼ');
b=histeq(a,256);
subplot(2,3,2);imshow(b);title('histeq����ֱ�Ӿ��⻯');
subplot(2,3,5);imhist(b);title('ֱ�Ӿ��⻯��ֱ��ͼ');

% I=a;
% [m,n]=size(I);
% h = zeros(1,256);
% %I=double(I);
% for i = 1:m
%   for j = 1:n
%     h(I(i,j)+1)=h(I(i,j)+1)+1;%ͳ��ԭʼͼ����Ҷȳ��ִ�������Ӧ�����h��
%     end
% end
% new=zeros(1,256);%����»Ҷ�ֵ����
% for i=1:256
%     temp=0;
%     for j=1:i
%         temp=temp+h(j);%������Ҷ�ֵ���ۼƷֲ�
%     end
%     new(i)=floor(temp*255/(m*n));%�����µĻҶ�ֵ
% end
% y=zeros(m,n);
% for i=1:m
%     for j=1:n
%         y(i,j)=new(I(i,j)+1);%���µĻҶ�ֵ�õ��µ�ͼ��
%     end
% end
% y1=uint8(y);%����ת��
% subplot(2,3,3);imshow(y1);title('���ʵ�־��⻯');
% subplot(2,3,6);imhist(y1);title('��̾��⻯��ֱ��ͼ');
