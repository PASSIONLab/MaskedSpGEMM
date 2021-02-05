function splittoproc(A, gridx, gridy, prefix)
%%
% Partition matrix A to a grid of dimensions (gridx) x (gridy)
% Write each submatrix into a file
% Written by : Aydin BULUC. 2021
%%




if nargin < 3
    error('Not enough arguments!');
    return
end

dimx = int64(size(A,1)/ gridx);
dimy = int64(size(A,2)/ gridy);

dirname = [prefix,'-proc',num2str(gridx),'-by-',num2str(gridy)];
if exist(dirname, 'dir') ~= 0
    rmdir(dirname,'s')
end
mkdir(dirname);
cd(dirname);

  
pdimx = int64(1);
pdimy = int64(1);
for i = 1:gridx-1
    for j = 1:gridy-1
        subA = A(pdimx:pdimx+dimx-1, pdimy:pdimy+dimy-1);
        mmwrite(['splitmatrix_',num2str(i),'_',num2str(j),'.mtx'], subA);
        pdimy = pdimy + dimy;
    end
    subA = A(pdimx:pdimx+dimx-1, pdimy:end);
    mmwrite(['splitmatrix_',num2str(i),'_',num2str(gridy),'.mtx'], subA);
    pdimx = pdimx + dimx;
    pdimy = 1;
end

for j = 1:gridy-1
    subA = A(pdimx:end, pdimy:pdimy+dimy-1);
    mmwrite(['splitmatrix_',num2str(gridx),'_',num2str(j),'.mtx'], subA);
    pdimy = pdimy + dimy;
end
subA = A(pdimx:end, pdimy:end);
mmwrite(['splitmatrix_',num2str(gridx),'_',num2str(gridy), '.mtx'], subA);
cd('..');
