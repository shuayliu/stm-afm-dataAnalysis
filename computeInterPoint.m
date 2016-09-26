function [ x,y ] = computeInterPoint( line1,line2 )
%计算线1和线2的交叉点，其中line1是直线，line2是斜线
%   Detailed explanation goes here
y=line1(1);
x=(y-line2(2))/line2(1);%即x=(y-b)/k;
end

