close all;clear all;clc
a=imread('C:\Users\HP\Desktop\File\tutor\summer\image\polyu database 3126ROI\1_1.bmp');
subplot(2,3,1);imshow(a);title('原图');
subplot(2,3,4);imhist(a);title('原图直方图');
b=histeq(a,256);
subplot(2,3,2);imshow(b);title('histeq函数直接均衡化');
subplot(2,3,5);imhist(b);title('直接均衡化后直方图');

% I=a;
% [m,n]=size(I);
% h = zeros(1,256);
% %I=double(I);
% for i = 1:m
%   for j = 1:n
%     h(I(i,j)+1)=h(I(i,j)+1)+1;%统计原始图像各灰度出现次数，对应存放在h中
%     end
% end
% new=zeros(1,256);%存放新灰度值个数
% for i=1:256
%     temp=0;
%     for j=1:i
%         temp=temp+h(j);%计算各灰度值的累计分布
%     end
%     new(i)=floor(temp*255/(m*n));%计算新的灰度值
% end
% y=zeros(m,n);
% for i=1:m
%     for j=1:n
%         y(i,j)=new(I(i,j)+1);%由新的灰度值得到新的图像
%     end
% end
% y1=uint8(y);%类型转换
% subplot(2,3,3);imshow(y1);title('编程实现均衡化');
% subplot(2,3,6);imhist(y1);title('编程均衡化后直方图');
