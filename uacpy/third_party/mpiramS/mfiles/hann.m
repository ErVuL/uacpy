function y=hann(x,T)
%HANN	Hann function
%	hann(x,T) returns a matrix whose elements are the hann window for the elements 
%	of X, i.e.
%	     y = sin^2(pi*x)/(T-1)   if x ~= 0
%	       = 1                   if x == 0
%	where x is an element of the input matrix and y is the resultant
%	output element.  
%
%            y = 0 for x < 0 or T < x


y=ones(size(x));
i=find(x);
y(i)=sin(pi*(x(i)+T/2)/(T-1));
y(i)=y(i).*y(i);

i=find(x<=-T/2 | T/2 < x);
y(i)=0*y(i);

