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
	for (int h = 0; h < imageHeight; h++)
	{
		for (int w = 0; w < imageWidth; w++)
		{	
				depth[h * imageWidth + w] = 0;

		}
	}
	
#pragma omp parallel for
	for(int y=featureHeight; y<imageHeight-featureHeight;y++)
		{
			for(int x=featureWidth; x<=imageWidth-featureWidth; x++)
		   {

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			/****************/
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					/* Skip feature boxes that dont fit in the displacement box. */
					if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;
					
					float squaredDiffer[4]={0.0,0.0,0.0,0.0};
					__m128 total = _mm_setzero_ps();
					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						for (int boxX = -featureWidth, i=1; i <= (2*featureWidth+1)-4; boxX+=4, i+=4)    //*************************************************
						{
							
							int leftX = x + boxX; 
							int leftY = y + boxY;
							int rightX = x + dx + boxX;
							int rightY = y + dy + boxY;

							__m128 left_row=_mm_loadu_ps(&left[leftY * imageWidth + leftX]);
							__m128 right_row=_mm_loadu_ps(&right[rightY * imageWidth + rightX]);
							__m128 difference = _mm_sub_ps(left_row, right_row);
							__m128 sqrtdiff=_mm_mul_ps(difference, difference);
							total=_mm_add_ps(total, sqrtdiff);
						}
					}

						_mm_storeu_ps(squaredDiffer, total);   //add
						squaredDifference+=squaredDiffer[0]+squaredDiffer[1]+squaredDiffer[2]+squaredDiffer[3];
						//without adding the extra, if already too large
						if (squaredDifference>minimumSquaredDifference && minimumSquaredDifference != -1) 
						{
							continue;
						}

						int leftY;
						int rightY;
						int k;
						if(featureWidth%2==0)
						{
							for(k=-featureHeight; k<=featureHeight; k++)
							{
								leftY=y+k;
								rightY=y+dy+k;
								float differ = left[ leftY* imageWidth + x+featureWidth] - right[ rightY* imageWidth + x+dx+featureWidth];
								squaredDifference += differ * differ;
							}
						}
						else{
							for(k=-featureHeight; k<=featureHeight; k++)
						{
							leftY=y+k;
							rightY=y+dy+k;
							int leftpos=leftY*imageWidth + x+ featureWidth;
							int rightpos=rightY *imageWidth+ x+dx+featureWidth;
							float differ_1 = left[leftpos] - right[rightpos];
							float differ_2 = left[leftpos-1] - right[ rightpos-1];
							float differ_3 = left[leftpos-2] - right[rightpos-2];
							squaredDifference += differ_1 * differ_1  + differ_2 * differ_2 +differ_3 * differ_3;
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

