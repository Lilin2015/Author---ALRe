function [W,record] = ALRe(P,I,R,maxIter,LB,UB)
%% input
% P, output image containing inconsistent edges
% I, input image
% R, side length of the square mask
% maxIter, allowed max iteration times
% LB and UB, lowbound of upperbound of residuals
%            pixels with sqrt(e) lower than LB are considered as inlier
%            pixels with sqrt(e) larger than UB are considered as outlier
%% output
% W, the fidelity map of image P
% record, details of the process, including:
% record.E, residual map
% record.DeltaE, maximum residual decrement of each iteration
% record.t, iteration times
%% params
if ~exist('LB','var'), LB=0.01; end
if ~exist('UB','var'), UB=0.3; end
%% prepare
[m,n,c] = size(I);
N = m*n;
Iv  = reshape(permute(I,[3,1,2]),[c,1,N]);
II  = bsxfun(@times,Iv,permute(Iv,[2,1,3]));
II  = reshape(permute(II,[3,1,2]),[m,n,c*c]);
IIv = reshape(permute(II,[3,1,2]),[c,c,N]);
Pv  = reshape(permute(P,[3,1,2]),[1,1,N]);
A   = zeros(c,1,N);
W   = ones(m,n);
Epre = ones(m,n);
DeltaE = zeros(maxIter,1);
%% iterations
for t = 1 : maxIter
    %% avg image
    mWMap = imboxfilt(W,R);
    mWP  = imboxfilt(W.*P,R);
    mWPP = imboxfilt(W.*P.*P,R);
    mWI  = imboxfilt(repmat(W,[1,1,c]).*I,R);
    mWPI = imboxfilt(repmat(W.*P,[1,1,c]).*I,R);
    %% avg cov-matrix
    mWII = imboxfilt(repmat(W,[1,1,c*c]).*II,R);
    %% into vectors
    % single channel
    mW    = reshape(permute(mWMap,[3,1,2]),[1,1,N]);
    mWP   = reshape(permute(mWP,[3,1,2]),[1,1,N]);
    mWPP  = reshape(permute(mWPP,[3,1,2]),[1,1,N]);
    % multi channel
    mWI   = reshape(permute(mWI,[3,1,2]),[c,1,N]);
    mWPI  = reshape(permute(mWPI,[3,1,2]),[c,1,N]);
    % matrix
    mWII  = reshape(permute(mWII,[3,1,2]),[c,c,N]);
    %% calculate parameters
    C = 0.001*eye(c)+mWII+bsxfun(@times,mW,IIv)-bsxfun(@times,mWI,permute(Iv,[2,1,3]))-bsxfun(@times,Iv,permute(mWI,[2,1,3]));
    d = mWPI-bsxfun(@times,Pv,mWI)-bsxfun(@times,mWP,Iv)+bsxfun(@times,mW.*Pv,Iv);
    for i = 1 : N, A(:,:,i) = C(:,:,i)\d(:,:,i); end
    B = Pv-sum(A.*Iv,1);
    E = sum(sum(repmat(A,[1,c,1]).*mWII,1).*permute(A,[2,1,3]),2)+mW.*B.^2+mWPP...
        +2*sum(bsxfun(@times,B,permute(A,[2,1,3])).*permute(mWI,[2,1,3]),2)...
        -2*sum(A.*mWPI,1)-2*B.*mWP;
    %% summary results
    E = reshape(permute(E,[3,1,2]),[m,n,1]);
    E(E<0)=0; E(E>1)=1; 
    E=E./mWMap; 
    E=E.^0.5;
    DeltaE(t) = max(max(abs(E-Epre))); Epre = E;
    if DeltaE(t)<0.01, break; end
    W = (1./min(max(E,LB),UB)-1/UB+0.01) / (1/LB-1/UB+0.01);
end
%% additional results
record.E = reshape(permute(E,[3,1,2]),[m,n,1]);
record.DeltaE = DeltaE;
record.t = t;
end

