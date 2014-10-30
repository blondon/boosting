function [wtw,prodmat] = kernelDotProd(examples1, examples2, alphas1, alphas2, mexKernelFunc, Z)

sameExamples = 0;
if isempty(examples2)
	examples2 = examples1;
	sameExamples = 1;
end
sameAlphas = 0;
if isempty(alphas2)
	alphas2 = alphas1;
	sameAlphas = 1;
end

[violators1,alphavecs1] = getViolatorsAlphas(examples1,alphas1);
if sameExamples && sameAlphas
	violators2 = violators1;
	alphavecs2 = alphavecs1;
else
	[violators2,alphavecs2] = getViolatorsAlphas(examples2,alphas2);
end

prodmat = zeros(length(examples1),length(examples2));

activeExamples1 = find(~cellfun(@isempty,alphas1));
activeExamples2 = find(~cellfun(@isempty,alphas2));
for idx1 = 1:length(activeExamples1)
	i = activeExamples1(idx1);
	for idx2 = 1:length(activeExamples2)
		j = activeExamples2(idx2);
		% Compute kernelized energy in mex function
		prodmat(i,j) = ...
			mexKernelFunc( ...
				int32(examples1(i).Y),int32(examples2(j).Y), ...
				int32(violators1{i}),int32(violators2{j}), ...
				examples1(i).edgeStruct.nStates,examples2(j).edgeStruct.nStates, ...
				examples1(i).edgeStruct.edgeEnds,examples2(j).edgeStruct.edgeEnds, ...
				examples1(i).Xnode,examples2(j).Xnode, ...
				examples1(i).Xedge,examples2(j).Xedge, ...
				alphavecs1{i},alphavecs2{j});
% 		prodmat(i,j) = ...
% 			linKernelDotProd( ...
% 				examples1(i).Y,examples2(j).Y, ...
% 				violators1{i},violators2{j}, ...
% 				examples1(i).edgeStruct,examples2(j).edgeStruct, ...
% 				examples1(i).Xnode,examples2(j).Xnode, ...
% 				examples1(i).Xedge,examples2(j).Xedge, ...
% 				alphavecs1{i},alphavecs2{j});
	end
end

% Normalize matrix
% (because sufficient stats are supposed to be normalized by Z)
prodmat = prodmat / Z^2;

% Sum of matrix elements is inner product w' * w
wtw = sum(prodmat(:));


% Collect worst violators and alphas
function [violators, alphavecs] = getViolatorsAlphas(examples, alphas)

violators = cell(length(examples),1);
alphavecs = cell(length(examples),1);
activeExamples = find(~cellfun(@isempty,alphas));
for idx = 1:length(activeExamples)
	i = activeExamples(idx);
	% Get current example and alphas
	ex_i = examples(i);
	alpha_i = alphas{i};
	% Collect states,alphas for worst violators
	ykeys = alpha_i.keys;
	alphavecs{i} = zeros(length(ykeys),1);
	violators{i} = zeros(length(ex_i.Y),length(ykeys));
	for yi = 1:length(ykeys)
		alpha_iy = alpha_i(ykeys{yi});
		alphavecs{i}(yi) = alpha_iy{1};
		violators{i}(:,yi) = alpha_iy{2};
	end
end


% MATLAB version
function wtw = linKernelDotProd(y1,y2,V1,V2,edgeStruct1,edgeStruct2, ...
	Xnode1,Xnode2,Xedge1,Xedge2,alphas1,alphas2)

% Normalizer
Z = length(y1) * length(y2);

% init output
wtw = 0;

% Compute node energy
for n1 = 1:length(y1)
	for n2 = 1:length(y2)
		if edgeStruct1.nStates(n1) == edgeStruct2.nStates(n2)
			ys1 = y1(n1);
			ys2 = y2(n2);
			for v1 = 1:size(V1,2)
				vs1 = V1(n1,v1);
				for v2 = 1:size(V2,2)
					vs2 = V2(n2,v2);
					% Determine how many times to add alpha1*alpha2*k(x,x')
					cnt = (vs1==vs2) + (ys1==ys2) - (vs1==ys2) - (ys1==vs2);
					if cnt ~= 0
						% Add kernel
						k = linearKernel(Xnode1(1,:,n1)',Xnode2(1,:,n2)') / Z;
						wtw = wtw + alphas1(v1) * alphas2(v2) * k * cnt;
					end
				end
			end
		end
	end
end

% Compute edge energy
for e1 = 1:edgeStruct1.nEdges
	n1_1 = edgeStruct1.edgeEnds(e1,1);
	n2_1 = edgeStruct1.edgeEnds(e1,2);
	for e2 = 1:edgeStruct2.nEdges
		n1_2 = edgeStruct2.edgeEnds(e2,1);
		n2_2 = edgeStruct2.edgeEnds(e2,2);
		if edgeStruct1.nStates(n1_1) == edgeStruct2.nStates(n1_2) && ...
		   edgeStruct1.nStates(n2_1) == edgeStruct2.nStates(n2_2)
			ys1_1 = y1(n1_1);
			ys2_1 = y1(n2_1);
			ys1_2 = y2(n1_2);
			ys2_2 = y2(n2_2);
			for v1 = 1:size(V1,2)
				vs1_1 = V1(n1_1,v1);
				vs2_1 = V1(n2_1,v1);
				for v2 = 1:size(V2,2)
					vs1_2 = V2(n1_2,v2);
					vs2_2 = V2(n2_2,v2);
					% Determine how many times to add alpha1*alpha2*k(x,x')
					cnt = ((vs1_1==vs1_2)&&(vs2_1==vs2_2)) ...
						+ ((ys1_1==ys1_2)&&(ys2_1==ys2_2)) ...
						- ((vs1_1==ys1_2)&&(vs2_1==ys2_2)) ...
						- ((ys1_1==vs1_2)&&(ys2_1==vs2_2));
					if cnt ~= 0
						% Add kernel
						k = linearKernel(Xedge1(1,:,e1)',Xedge2(1,:,e2)') / Z;
						wtw = wtw + alphas1(v1) * alphas2(v2) * k * cnt;
					end					
				end
			end
		end
	end
end


