function cvrmse = wrapFitNet(x, y, varargin)
% Handle variable inputs
ip = inputParser;
ip.addParameter('hiddenLayerSize', 20);
ip.addParameter('CVPartition', cvpartition(numel(y),'Holdout', 0.10));
parse(ip, varargin{:});
cv = ip.Results.CVPartition;
hiddensz = ip.Results.hiddenLayerSize;
% Train net.  You would adjust other hyper parameters here.
net = fitnet(hiddensz);
nets = train(net, x(:, cv.training.'), y(:, cv.training.'));
% Evaluate on test set and compute rmse
ypred = nets(x(:, cv.test.'));
cvrmse = sqrt(mean((ypred-y(cv.test)).^2));
end