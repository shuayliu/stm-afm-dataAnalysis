function [ x,y ] = computeInterPoint( line1,line2 )
%������1����2�Ľ���㣬������һ��ֱ�ߣ���2��б��
%   Detailed explanation goes here
y=line1(1);
x=(y-line2(2))/line2(1);%��x=(y-b)/k;
end
