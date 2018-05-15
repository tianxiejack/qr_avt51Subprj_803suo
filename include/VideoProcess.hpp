
#ifndef VIDEOPROCESS_HPP_
#define VIDEOPROCESS_HPP_

#include "MultiChVideo.hpp"
#include "Displayer.hpp"
#include "UtcTrack.h"
#include "multitarget.h"
#include "BlobDetector.h"
#include "app_osdgrp.h"
#include "app_status.h"
#include"osd_graph.h"

#include "MMTD.h"
#include "mvdetectInterface.hpp"

typedef struct _main_thr_obj_cxt{
	bool bTrack;
	bool bMtd;
	bool bBlobDetect;
	bool bMoveDetect;
	int chId;
	int iTrackStat;
	
	//Mat frame;
}MAIN_ProcThrObj_cxt;

typedef struct _main_thr_obj{
	MAIN_ProcThrObj_cxt cxt[2];
	OSA_ThrHndl		thrHandleProc;
	OSA_SemHndl	procNotifySem;
	int pp;
	bool bFirst;
	volatile bool	exitProcThread;
	bool						initFlag;
	void 						*pParent;
}MAIN_ProcThrObj;


typedef struct _Track_info{
	UTC_RECT_float trackrect;
	UTC_RECT_float reAcqRect;
	unsigned int  trackfov;
	unsigned int TrkStat;

}Track_InfoObj;


class CVideoProcess
{
	MAIN_ProcThrObj	mainProcThrObj;
	Mat mainFrame[2];
public:
	CVideoProcess();
	~CVideoProcess();
	int creat();
	int destroy();
	int init();
	typedef enum{
		VP_CFG_MainChId = CDisplayer::DS_CFG_Max,
		VP_CFG_SubChId,
		VP_CFG_TrkEnable,
		VP_CFG_MtdEnable,
		VP_CFG_BlobEnable,
		VP_CFG_SubPicpChId,
		VP_CFG_MvDetect,
		VP_CFG_Max
	}VP_CFG;
	int dynamic_config(int type, int iPrm, void* pPrm = NULL);
	int run();
	int stop();

public:
	virtual void OnCreate(){};
	virtual void OnDestroy(){};
	virtual void OnInit(){};
	virtual void OnConfig(){};
	virtual void OnRun(){};
	virtual void OnStop(){};
	virtual void Ontimer(){};
	virtual bool OnPreProcess(int chId, Mat &frame){return true;}
	virtual bool OnProcess(int chId, Mat &frame){return true;}
	virtual void OnMouseLeftDwn(int x, int y){};
	virtual void OnMouseLeftUp(int x, int y){};
	virtual void OnMouseRightDwn(int x, int y){};
	virtual void OnMouseRightUp(int x, int y){};
	virtual void OnKeyDwn(unsigned char key){};

	int m_ImageAxisx;
	int m_ImageAxisy;
	int m_SensorStat;

public :

	//CMSTracker trackInit;
	int detState ;
	int trackEnd ;
	int trackStart ;
	bool nextDetect;
	int lastFrameBox ;
	int prichnalid;
	//vector<Rect> Box;
	bool moveStat;
	bool algOsdRect;
	bool moveDetectRect;
	
	bool TrkAim43;
	bool wFileFlag;
	bool tvzoomStat;

	ALGMTD_HANDLE m_mtd[MAX_CHAN];
	
protected:
	MultiChVideo MultiCh;
	//BigChVideo		BigChannel;
	CDisplayer m_display;
	Mat m_dc;
	Mat m_dccv;

	int adaptiveThred;

	UTCTRACK_HANDLE m_track;

	
	static bool m_bTrack;
	static bool m_bMtd;
	static bool m_bBlobDetect;
	static bool m_bMoveDetect;
	static int m_iTrackStat;
	static int m_iTrackLostCnt;
	
	Uint32 rememtime;
	bool rememflag;
	int m_curChId;
	int m_curSubChId;
	int trackchange;
	int m_searchmod;

	int Enhmod;
	float Enhparm;
	int DetectGapparm;
	int MinArea;
	int MaxArea;
	int stillPixel;
	int movePixel;
	float lapScaler;
	int lumThred;

	FILE *m_pwFile;

	int configEnhFromFile();
	int configAvtFromFile();
	int Algconfig();
	
	void process_event(int type, int iPrm, void *pPrm);
	int process_frame(int chId, Mat frame);
	int process_mtd(ALGMTD_HANDLE pChPrm, Mat frame_gray, Mat frame_dis);

	#if __TRACK__
	Track_InfoObj *trackinfo_obj;
	int process_track(int trackStatus, Mat frame_gray, Mat frame_dis, UTC_RECT_float &rcResult);
	int ReAcqTarget();
	void Track_reacq(UTC_RECT_float & m_rcTrack,int acqinterval);
	void Track_fovreacq(int fov,int sensor,int sensorchange);
	#endif
	

	static int m_mouseEvent, m_mousex, m_mousey;
	static CVideoProcess *pThis;
	static void call_run(int value);
	static int callback_process(void *handle, int chId, Mat frame);
	static void mouse_event(int button, int state, int x, int y);
	static void keyboard_event(unsigned char key, int x, int y);
	static void keySpecial_event( int key, int x, int y);
	static void visibility_event(int state);
	static void close_event(void);

private:
	OSA_MutexHndl m_mutex;
//	unsigned char *m_grayMem[2];
	char m_strDisplay[128];
	void main_proc_func();
	int MAIN_threadCreate(void);
	int MAIN_threadDestroy(void);
	static void *mainProcTsk(void *context)
	{
		MAIN_ProcThrObj  * pObj= (MAIN_ProcThrObj*) context;
		if(pObj==NULL)
			{

			OSA_printf("++++++++++++++++++++++++++\n");

			}
		CVideoProcess *ctxHdl = (CVideoProcess *) pObj->pParent;
		ctxHdl->main_proc_func();
		OSA_printf("****************************************************\n");
		return NULL;
	}
	static void extractYUYV2Gray2(Mat src, Mat dst);
	static int64 tstart;

protected: //track
	UTC_RECT_float m_rcTrack, m_rcAcq;
	UTC_Rect preAcpSR;
	UTC_Rect MoveAcpSR;
	UTC_Rect TRKMoveAcpSR;
	BlobRect		    m_blobRect;
	int			    m_intervalFrame;
	int			  m_intervalFrame_change;
	int 			    m_bakChId;

	CMMTD	m_MMTDObj;
	TARGETBOX	m_tgtBox[MAX_TARGET_NUMBER];	

#if __MOVE_DETECT__
	#if __DETECT_SWITCH_Z__
		protected:
		CMvDectInterface *m_pMovDetector;
		void	initMvDetect();
		void	DeInitMvDetect();
		static void NotifyFunc(void *context, int chId);
		std::vector<TRK_RECT_INFO> detect_vect;
		std::vector<TRK_RECT_INFO> detect_bak;
	#else
		protected:
		cv::Rect boundRect[6];
		cv::Rect backRect[6];
	#endif
#endif


};



#endif /* VIDEOPROCESS_HPP_ */
