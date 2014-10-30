function energy = computeEnergy(y, nodePot, edgePot, edgeStruct)
%
% Computes the energy of a given configuration y, from node/edge potentials
%
energy = 0;
for n = 1:edgeStruct.nNodes
	energy = energy + log(nodePot(n,y(n)));
end
for e = 1:edgeStruct.nEdges
	n1 = edgeStruct.edgeEnds(e,1);
	n2 = edgeStruct.edgeEnds(e,2);
	energy = energy + log(edgePot(y(n1),y(n2),e));
end
