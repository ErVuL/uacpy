function y=sinc(x)
%SINC	Sin(pi*x)/(pi*x) function
%	SINC(X) returns a matrix whose elements are the sinc of the elements 
%	of X, i.e.
%	     y = sin(pi*x)/(pi*x)    if x ~= 0
%	       = 1                   if x == 0
%	where x is an element of the input matrix and y is the resultant
%	output element.  

%	Author(s): T. Krauss, 1-14-93
%	Copyright (c) 1984-94 by The MathWorks, Inc.
%       $Revision: 1.4 $  $Date: 1994/01/25 17:59:49 $

y=ones(size(x));
i=find(x);
y(i)=sin(pi*x(i))./(pi*x(i));
