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
	// invalid write of size
	int b;
	__m128 zero=_mm_setzero_ps();
	//resevred. #pragma omp parallel for
	for (int w = 0; w < imageWidth*imageHeight/4*4; w+=4)
	{	
			_mm_storeu_ps(&depth[w], zero);
	}
		//tail case:
	for (b = imageWidth*imageHeight/4*4;  b< imageWidth*imageHeight; b++)
	{	
			depth[b] = 0;
	}

	int even=0;
	if(featureHeight%2==0)
	{				
		even=1;
	}

#pragma omp parallel for collapse(2)
	for(int y=featureHeight; y<=imageHeight-featureHeight-1;y++)
		{
			for(int x=featureWidth; x<=imageWidth-featureWidth-1; x++)
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
					
					float squaredDiffer[4];
					float temp[4]={0,0,0,0};
					__m128 tempp;
					__m128 total = _mm_setzero_ps();
					__m128 left_row;
					__m128 right_row;
					__m128 difference;
					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					int padding=(2*featureWidth+1+1)-4;
					if(even)
					{
						padding=(2*featureWidth+1)-4;
					}

					
					for (int boxX = -featureWidth, i=0; i <= padding; boxX+=4, i+=4) 
					{
							int leftX = x + boxX; 
							int rightX = x + dx + boxX;
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)   //*************************************************
						{
							
							int leftY = y + boxY;
							int rightY = y + dy + boxY;
							left_row=_mm_loadu_ps(&left[leftY * imageWidth + leftX]);
							right_row=_mm_loadu_ps(&right[rightY * imageWidth + rightX]);
							if(even==0 && i==padding)
							{
								temp[3]=left[leftY * imageWidth + leftX+ 3];
								tempp=_mm_loadu_ps(&temp[0]);
								left_row=_mm_sub_ps(left_row, tempp);
								temp[3]=right[rightY * imageWidth + rightX+ 3];
								tempp=_mm_loadu_ps(&temp[0]);
								right_row=_mm_sub_ps(right_row, tempp);
							}
							difference = _mm_sub_ps(left_row, right_row);
							difference=_mm_mul_ps(difference, difference);
							total=_mm_add_ps(total, difference);
						}
					}

						_mm_storeu_ps(squaredDiffer, total);   //save to array.
						squaredDifference+=squaredDiffer[0]+squaredDiffer[1]+squaredDiffer[2]+squaredDiffer[3];
						//without adding the extra, if already too large
						if (squaredDifference>minimumSquaredDifference && minimumSquaredDifference != -1) 
						{
							continue;
						}

						int leftY;
						int rightY;
						int k;
						if(even==1)
						{
							for(k=-featureHeight; k<=featureHeight; k++)
							{
								leftY=y+k;
								rightY=y+dy+k;
								float differ = left[ leftY* imageWidth + x+featureWidth] - right[ rightY* imageWidth + x+dx+featureWidth];
								squaredDifference += differ * differ;
							}
						}
						//no need to odd special case.

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
				if (maximumDisplacement != 0)
				{
					depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				}
		}
		}
	}