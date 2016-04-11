// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	/* The two outer for loops iterate through each pixel */
	//depth array size= imageheight * imagewidth
	#pragma omp parallel for
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
		{	
			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
			if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
			{
				depth[y * imageWidth + x] = 0;
				continue;
			}

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			/****************/
			int startingY=-maximumDisplacement; 
			int startingX=-maximumDisplacement;
			int endY=maximumDisplacement;
			int endX=maximumDisplacement;
			if(y-maximumDisplacement-featureHeight<0)
			{
				startingY=featureHeight-y;
			}
			if(x-maximumDisplacement-featureWidth<0)
			{
				startingX=featureWidth-x;
			}
			if(y + maximumDisplacement + featureHeight >= imageHeight)
			{
				endY=imageHeight-featureHeight-y-1;
			}
			if(x + maximumDisplacement + featureWidth >= imageHeight)
			{
				endX=imageWidth-featureWidth-x-1;
			}

			for (int dy = startingY; dy <= endY; dy++)
			{	
				for (int dx = startingX; dx <= endX; dx++)
				{

					float squaredDifference = 0;

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						for (int boxX = -featureWidth, i=1; i <= (2*featureWidth+1)-4; boxX+=4, i+=4)    //*************************************************
						{
							
							int leftX = x + boxX; 
							int leftY = y + boxY;
							int rightX = x + dx + boxX;
							int rightY = y + dy + boxY;

							float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
							squaredDifference += difference * difference;

							difference = left[leftY * imageWidth + leftX+1] - right[rightY * imageWidth + rightX+1];
							squaredDifference += difference * difference;

							difference = left[leftY * imageWidth + leftX+2] - right[rightY * imageWidth + rightX+2];
							squaredDifference += difference * difference;

							difference = left[leftY * imageWidth + leftX+3] - right[rightY * imageWidth + rightX+3];
							squaredDifference += difference * difference;
						
						}
						//without adding the extra, if already too large
						if (squaredDifference>minimumSquaredDifference && minimumSquaredDifference != -1) 
						{
							continue;
						}

						int leftY = y + boxY;
						int rightY = y + dy + boxY;
						if(featureWidth%2==0)
						{
							float differ = left[ leftY* imageWidth + x+featureWidth] - right[ rightY* imageWidth + x+dx+featureWidth];
							squaredDifference += differ * differ;
						}
						else{
							int leftpos=leftY*imageWidth + x+ featureWidth;
							int rightpos=rightY *imageWidth+ x+dx+featureWidth;
							float differ_1 = left[leftpos] - right[rightpos];
							float differ_2 = left[leftpos-1] - right[ rightpos-1];
							float differ_3 = left[leftpos-2] - right[rightpos-2];
							squaredDifference += differ_1 * differ_1  + differ_2 * differ_2 +differ_3 * differ_3 ;
						}
					}

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[y * imageWidth + x] = 0;
				}
				else
				{
					depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				}
			}
			else
			{
				depth[y * imageWidth + x] = 0;
			}
  		}
	}
}
