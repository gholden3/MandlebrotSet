/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include "Complex.cu"
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <math.h>
#include <GL/glext.h>
#include <GL/gl.h>
#include <GL/glu.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512
#define NUM_THREADS 32
using namespace std;

// Initial screen coordinates, both host and device.
// x goes from -2 to 1 and y oges from -1.2 to 1.8
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
int* dev_MArr;
int MArr[512*512];
const int maxIt = 2000; // Msximum Iterations

double coordinateReal(double x){//calculate real part of coordinate from x value
return (minC.r + x*(maxC.r-minC.r)/(511));
}

double coordinateImaginary(double y){
return (maxC.i - y*(maxC.i-minC.i)/(511));
}
// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors()
{
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 5)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
          colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}

__global__ void calculateMandlebrot(Complex* dev_minC, Complex* dev_maxC, int* dev_MArr){
  int n;
  int id = threadIdx.x + blockIdx.x * blockDim.x; //what pixel should I work on
  int x = id%512; //what is x position
  int y = id/512; //what is y position
  double coordImaginary = (dev_maxC->i) - y*((dev_maxC->i)-(dev_minC->i))/(511);
  double coordReal = (dev_minC->r) + x*((dev_maxC->r)-(dev_minC->r))/(511);
  Complex cCoord = Complex(coordReal, coordImaginary);
  Complex zCoord = Complex(coordReal, coordImaginary);
  for(n=0;n<maxIt;++n){
    if(zCoord.magnitude2() >  4.0)
      break;
    zCoord = (zCoord * zCoord) + cCoord;
  }  
  dev_MArr[id] = n;
/*for( int  y=0; y<WINDOW_DIM; ++y){
    double coordIm = coordinateImaginary(y);
    for( int x=0;  x<WINDOW_DIM; ++x){
      double coordReal = coordinateReal(x);
      Complex cCoord = Complex(coordReal,coordIm);
      bool isInside = true;
      Complex zCoord = Complex(coordReal,coordIm);
      for( n=0;n<=maxIt;++n){//calculate whether it is in the set
	if(zCoord.magnitude2() > 2.0)
	  {
	    isInside = false;
	    break;
	  }
	// z = z^2 + c
	zCoord = (zCoord * zCoord) + cCoord;
      }
      MArr[x][y] = n; //set array point to iteration count.
    }
  }*/
}

void displayMandlebrot(){
  //cout << "hello from displayMandlebrot" << endl;
  glBegin(GL_POINTS);
  for(int x =0; x< WINDOW_DIM; x++){
    for(int y=0; y<WINDOW_DIM; y++){
      //draw it based on iteration
      int pix = y*512 + x;
      int iterationCount = MArr[pix];
      glColor3f(colors[iterationCount].r,colors[iterationCount].g,colors[iterationCount].b);
      glVertex2f(x,y);  
    }
  }
  glEnd();
}
void display(void){
  //cout << "hello from display ()" << endl;
  glClear(GL_COLOR_BUFFER_BIT);
  glClear(GL_DEPTH_BUFFER_BIT);
  displayMandlebrot();
  glutSwapBuffers();
}

void init(){
  glShadeModel(GL_FLAT);
  glViewport(0,0,WINDOW_DIM, WINDOW_DIM);
  //drawMandlebrot();
}

void getReadyForCalcMandlebrot(){
  //allocate space for device copies
  cudaMalloc((void**)&dev_MArr, WINDOW_DIM * WINDOW_DIM * sizeof(int));
  cudaMalloc((void**)&dev_minC, sizeof(Complex));
  cudaMalloc((void**)&dev_maxC, sizeof(Complex));
  //copy inputs to device
  cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
  //launch calculateMandlebrot() kernel
  calculateMandlebrot<<<WINDOW_DIM * WINDOW_DIM / NUM_THREADS, NUM_THREADS >>>(dev_minC, dev_maxC, dev_MArr);
  //copy result back to host
  cudaMemcpy(MArr, dev_MArr, WINDOW_DIM * WINDOW_DIM * sizeof(int), cudaMemcpyDeviceToHost);
  //free
  cudaFree(dev_minC); cudaFree(dev_maxC); cudaFree(dev_MArr);
}


int main(int argc, char** argv)
{ getReadyForCalcMandlebrot();
  // Initialize OPENGL here
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(WINDOW_DIM,WINDOW_DIM);
  glutInitWindowPosition(100,100);
  glutCreateWindow("Mandlebrot");
  init();
  glViewport(0,0, (GLsizei) 512, (GLsizei) 512);
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, WINDOW_DIM, 0, WINDOW_DIM, -WINDOW_DIM, WINDOW_DIM);
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard
  glutDisplayFunc(display);
  glutIdleFunc(display);
  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels
  InitializeColors();
  glutMainLoop(); // THis will callback the display, keyboard and mouse
  return 0;
  
}
