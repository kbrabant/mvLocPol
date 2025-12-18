function [Hopt,loocv] = cvlocpolOctave(X, y, deg, kernel, Hinit)

% pkg load optim
options = optimset('Display','off','Algorithm','interior-point');
%options = optimset('Display','off');

if isscalar(Hinit)
    LB = 2/size(X,1); UB = 4;
else
    LB =[2/size(X,1) 0; 0 2/size(X,1)]; UB =[5 eps; eps 5];
end
[Hopt, loocv] = fmincon(@(H)loolocpol(X,y,deg,kernel,H),Hinit,[],[],[],[],LB,UB,[],options);
%[Hopt, loocv] = fminsearch(@(H)loolocpol(X,y,deg,kernel,H),Hinit,options);

end

function cv = loolocpol(X,y,deg,kernel,H)
[Yh,L] = mvlocpolOctave(X,y, deg, kernel, H);
cv = (1/size(X,1))*sum(((y-Yh)./(1-diag(L))).^2); % LOO-CV
%cv = (1/size(X,1))*((y-Yh)'*(y-Yh)); % RSS
end

