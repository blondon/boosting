#include <math.h>
#include "mex.h"
#include "kernels.h"

/**
 * Computes (accumulates) the potentials for linear kernel StructSVM.
 *
 * INPUTS
 * y
 * V
 * nodePot
 * edgePot
 * nStates1,nStates2
 * edgeEnds1,edgeEnds2
 * Xnode1,Xnode2
 * Xedge1,Xedge2
 * alphas
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Input variables */
	int *y;
	int *V;
	double *nodePot;
	double *edgePot;
	int *nStates1, *nStates2;
	int *edgeEnds1, *edgeEnds2;
	double *Xnode1, *Xnode2;
	double *Xedge1, *Xedge2;
	double *alphas;
	
	/* Local variables */
	int nNodes1, nNodes2;
	int nEdges1, nEdges2;
	int nNodeFeats;
	int nEdgeFeats;
	int maxState;
	int nViolators;
	int n1, n2, n1_1, n2_1, n1_2, n2_2;
	int e1, e2;
	int s1, s2;
	int v;
	double k;
	double Z;
	double alphasum;
	
	/* Input */
	y = (int*)mxGetPr(prhs[0]);
	V = (int*)mxGetPr(prhs[1]);
	nodePot = (double*)mxGetPr(prhs[2]);
	edgePot = (double*)mxGetPr(prhs[3]);
	nStates1 = (int*)mxGetPr(prhs[4]);
	nStates2 = (int*)mxGetPr(prhs[5]);
	edgeEnds1 = (int*)mxGetPr(prhs[6]);
	edgeEnds2 = (int*)mxGetPr(prhs[7]);
	Xnode1 = (double*)mxGetPr(prhs[8]);
	Xnode2 = (double*)mxGetPr(prhs[9]);
	Xedge1 = (double*)mxGetPr(prhs[10]);
	Xedge2 = (double*)mxGetPr(prhs[11]);
	alphas = (double*)mxGetPr(prhs[12]);
	
	/* Dimensions */
	nNodes1 = mxGetDimensions(prhs[0])[0];
	nNodes2 = mxGetDimensions(prhs[2])[0];
	nEdges1 = mxGetDimensions(prhs[6])[0];
	nEdges2 = mxGetDimensions(prhs[7])[0];
	nNodeFeats = mxGetDimensions(prhs[8])[1];
	nEdgeFeats = mxGetDimensions(prhs[10])[1];
	maxState = mxGetDimensions(prhs[2])[1];
	nViolators = mxGetDimensions(prhs[1])[1];
	
	/* Normalize the kernel by (nNodes1 * nNodes2) */
	Z = (double)nNodes1;
	
	/* Compute sum of alphas */
	alphasum = 0.0;
	for (v = 0; v < nViolators; v++)
	{
		alphasum += alphas[v];
	}

	/* Compute nodePot */
	for (n2 = 0; n2 < nNodes2; n2++)
	{
		for (n1 = 0; n1 < nNodes1; n1++)
		{
			/* Determine if nodes of same type */
			if (nStates1[n1] == nStates2[n2])
			{
				/* Compute kernel between features */
				k = linearKernel(Xnode1,Xnode2,nNodeFeats,n1,n2) / Z;
				
				/* Update potentials for true assignment */
				s1 = y[n1]-1;
				nodePot[n2+nNodes2*s1] += alphasum * k;
				
				/* Update potentials for each violator */
				for (v = 0; v < nViolators; v++)
				{
					s1 = V[n1+nNodes1*v]-1;
					nodePot[n2+nNodes2*s1] -= alphas[v] * k;
				}
			}
		}
	}

	/* Compute edgePot */
	for (e2 = 0; e2 < nEdges2; e2++)
	{
		n1_2 = edgeEnds2[e2]-1;
		n2_2 = edgeEnds2[e2+nEdges2]-1;
		
		for (e1 = 0; e1 < nEdges1; e1++)
		{
			n1_1 = edgeEnds1[e1]-1;
			n2_1 = edgeEnds1[e1+nEdges1]-1;

			/* Determine if edges are of same type */
			if ((nStates1[n1_1] == nStates2[n1_2]) && (nStates1[n2_1] == nStates2[n2_2]))
			{
				/* Compute kernel between features */
				k = linearKernel(Xedge1,Xedge2,nEdgeFeats,e1,e2) / Z;
				
				/* Update potentials for true assignment */
				s1 = y[n1_1]-1;
				s2 = y[n2_1]-1;
				edgePot[s1+maxState*(s2+maxState*e2)] += alphasum * k;
				
				/* Update potentials for each violator */
				for (v = 0; v < nViolators; v++)
				{
					s1 = V[n1_1+nNodes1*v]-1;
					s2 = V[n2_1+nNodes1*v]-1;
					edgePot[s1+maxState*(s2+maxState*e2)] -= alphas[v] * k;
				}
			}
		}
	}
}