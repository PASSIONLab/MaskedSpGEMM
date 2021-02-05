function [c,triangles,triples,T1] = ccoeff_ll(A,verbose)
% CCOEFF : clustering coefficient of graph 
%
% c = ccoeff(A)
% [c,triangles,triples] = ccoeff(A)
%
% A is the adjacency matrix of a graph
%   (edge directions and self-loops are ignored)
% c is the clustering coefficient,
%   c = 3*triangles / triples, always between 0 and 1
% triangles is the number of 3-vertex subgraphs that are K3's
% triples is the number of 3-vertex subgraphs that are paths
%
% ... = ccoeff(A,1) prints some extra information
% 
% John R. Gilbert, 3 May 2011
% Modified by Aydin Buluc so it does 2X less compute, Jan 27, 2021

A = A - diag(diag(A));
A = double(A|A');

% Count triples by middle vertex
degree = full(sum(A));
triples = sum(degree.*(degree-1)/2);

% Sort by degree for more efficient triangle counting
[degree,p] = sort(degree);
A = A(p,p);

% Count each triangle by middle numbered vertex
T1 = tril(A); % T1(i,j)~=0: i-j in G and i>j
% also T1(j,k)~=0: j-k in G and j>k
B = T1*T1;    % B(i,k): count of j with i-j-k in G with i>j>k
triangles = sum(full(sum(T1 .* B)));  % count of such i-j-k closed by i-k

           
           
coeff = 3*triangles/triples;
if nargin>1 || nargout==0
    nnzA = nnz(A)
    nnzB = nnz(B)
    nnzC = nnz(T1 .* B)
    triangles
    triples
    coeff
end;
if nargout>=1
    c = coeff;
end;


            

