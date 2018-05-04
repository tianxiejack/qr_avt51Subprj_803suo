
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <glut.h>
#include "process021.hpp"
#include "dx_main.h"
#include "Ipc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	CProcess021 proc;
#ifdef __IPC__
	Ipc_pthread_start();
#endif
	App_dxmain();
	proc.creat();
	proc.init();
	proc.run();
	glutMainLoop();
	proc.destroy();
#ifdef __IPC__
	Ipc_pthread_stop();
#endif
    	return 0;
}

//__MMT__
//__BLOCK__
//__TRACK__