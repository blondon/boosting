function ss = sufficientStats(y, Xnode, Xedge, nodeMap, edgeMap, edgeStruct)

ss = zeros(edgeStruct.nParams,1);

for n = 1:edgeStruct.nNodes
	for f = 1:edgeStruct.nNodeFeats(n)
		p = nodeMap(n,y(n),f);
		ss(p) = ss(p) + Xnode(1,f,n);
	end
end

for e = 1:edgeStruct.nEdges
	for f = 1:edgeStruct.nEdgeFeats(e)
		n1 = edgeStruct.edgeEnds(e,1);
		n2 = edgeStruct.edgeEnds(e,2);
		p = edgeMap(y(n1),y(n2),e,f);
		ss(p) = ss(p) + Xedge(1,f,e);
	end
end