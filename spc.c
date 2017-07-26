#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "spc.h"

#define  NPROCS   288

#define  EPS    2.220446e-16

#define ORIGINAL 0

double  A[N][N];
double  b[M][N];
double  x[M][N];
double  c[N];
double  ctmp[M][N];


int     myid, numprocs;


void spc(double [N][N], double [M][N], double [M][N], int, int); 

void main(int argc, char* argv[]) {

     double  t0, t1, t2, t_w;
     double  dc_inv, d_mflops, dtemp, dtemp2, dtemp_t;

     int     ierr;
     int     i, j;
     int     ii;      
     int     ib;

     ierr = MPI_Init(&argc, &argv);
     ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
     ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);



      /* matrix generation --------------------------*/
      for(j=0; j<N; j++) {
        ii = 0;
        for(i=j; i<N; i++) {
          A[j][i] = (N-j) - ii;
          A[i][j] = A[j][i];
          ii++;
        }
      }
      /* end of matrix generation -------------------------- */

     /* set vector b  -------------------------- */
      for (i=0; i<N; i++) {
        b[0][i] = 0.0;
        for (j=0; j<N; j++) {
          b[0][i] += A[i][j];
        }
      }
      for (i=0; i<M; i++) {
        for (j=0; j<N; j++) {
          b[i][j] = b[0][j];
        }
      }
     /* ----------------------------------------------------- */


     /* Start of spc routine ----------------------------*/
     ierr = MPI_Barrier(MPI_COMM_WORLD);
     t1 = MPI_Wtime();

     spc(A, b, x, N, M);

     //ierr = MPI_Barrier(MPI_COMM_WORLD);
     t2 = MPI_Wtime();
     t0 =  t2 - t1; 
     ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     /* End of spc routine --------------------------- */

     if (myid == 0) {

       printf("--------------------------- \n");
       printf("N = %d , M = %d \n",N,M);
       printf("LU solve time  = %lf [sec.] \n",t_w);

       d_mflops = 2.0/3.0*(double)N*(double)N*(double)N;
       d_mflops += 7.0/2.0*(double)N*(double)N;
       d_mflops += 4.0/3.0*(double)N*(double)M;
       d_mflops = d_mflops/t_w;
       d_mflops = d_mflops * 1.0e-6;
       printf(" %lf [MFLOPS] \n", d_mflops);

     }

     /* Verification routine ----------------- */
     ib = N / NPROCS;
     dtemp_t = 0.0;
     for(i=0; i<M; i++) {
       for(j=myid*ib; j<(myid+1)*ib; j++) {
         dtemp2 = x[i][j] - 1.0;
         dtemp_t += dtemp2*dtemp2;
       }
     }
     dtemp_t = sqrt(dtemp_t);
     /* -------------------------------------- */

     MPI_Reduce(&dtemp_t, &dtemp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

     /* do not modify follows. -------- */ 
     if (myid == 0) {
       dtemp2 = (double)N*(double)N;      
       dtemp_t = EPS*(double)N*dtemp2*(double)M;
       dtemp_t = sqrt(dtemp_t);
       printf("Pass value: %e \n", dtemp_t);
       printf("Calculated value: %e \n", dtemp);
       if (dtemp > dtemp_t) {
          printf("Error! Test is falled. \n");
          exit(1);
       } 
       printf(" OK! Test is passed. \n");
       printf("--------------------------- \n");
     }
     /* ----------------------------------------- */


     ierr = MPI_Finalize();

     exit(0);
}


void spc(double A[N][N], double b[M][N], double x[M][N], int n, int m) 
{
  int i, j, k, ne;
     double dtemp, dtemp2;
     int istart, iend;
     int ib = n / numprocs;
     MPI_Status istatus;
     int idiagPE;


     if(!ORIGINAL){
       /* LU decomposition ---------------------- */
       istart = myid * ib;
       iend = (myid + 1) * ib;
       int isendPE;

       for (k=0; k<iend; k++) {
	 double buf[n-k-1]; // 通信用のバッファ
	 idiagPE = k / ib;
	 if (myid == idiagPE){// 枢軸ベクトルを持つPE
	   dtemp = 1.0 / A[k][k];
	   for (i=k+1; i<n; i++) {
	     dtemp2 = A[i][k]*dtemp;
	     A[i][k] = dtemp2;
	     buf[i-k-1] = dtemp2;
	   }
	   if(myid != numprocs-1)
	     MPI_Send(buf, n-k-1, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
	   istart = k+1;
	 } else {// 枢軸ベクトルを持たないPE
	   MPI_Recv(buf, n-k-1, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD, &istatus);
	   if(myid != numprocs-1)
	     MPI_Send(buf, n-k-1, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
	 }

	 for (i=k+1; i<n; i++) {
	   dtemp = buf[i-k-1];
	   A[i][k] = dtemp;
	   for (j=istart; j<iend; j++) {
	     A[i][j] = A[i][j] - A[k][j]*dtemp;
	   }
	 }
       }

       MPI_Barrier(MPI_COMM_WORLD);
       /* --------------------------------------- */


       int kk;


       /* Forward substitution ------------------ */

       istart = myid * ib;
       iend = (myid + 1) * ib;

       /* cをまとめた配列の初期化 */
       for (ne=0; ne<m; ne++)
	 for (k=0; k<n; k++)
	   ctmp[ne][k] = 0.0;

       double cbuf[m][ib];
       int ictmp, jctmp;

       for (k=0; k<n; k+=ib) {
	 /* 通信用の配列の初期化 */
	 for(ictmp=0; ictmp<m; ictmp++)
	   for(jctmp=0; jctmp<ib; jctmp++)
	     cbuf[ictmp][jctmp] = 0;

	 if (k >= istart) {/* 担当するブロックがある */
	   /* 左隣のPEからデータを受け取る */
	   if(myid != 0){
	     MPI_Recv(cbuf, ib*m, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD, &istatus);
	   }

	   idiagPE = k / ib;
	   if(myid == idiagPE){/* 対角ブロックを持つPE */
	     for(ne=0; ne<m; ne++){
	       for(kk=0; kk<ib; kk++){
		 ctmp[ne][k+kk] = cbuf[ne][kk] + b[ne][k+kk];
		 for(j=istart; j<istart+kk; j++)
		   ctmp[ne][k+kk] -= A[k+kk][j] * ctmp[ne][j];
	       }
	     }
	   } else {/* 対角ブロックを持たないPE */
	     /* 自分の所有範囲のデータのみ計算（まだ最終結果ではない） */
	     for(ne=0; ne<m; ne++)
	       for(kk=0; kk<ib; kk++)
		 for(j=istart; j<iend; j++){
		   cbuf[ne][kk] -= A[k+kk][j] * ctmp[ne][j];
		   ctmp[ne][k+kk] = cbuf[ne][kk];
		 }

	     /* 右隣のPEに，自分の担当範囲のデータを用いた演算結果を送る */
	     if(myid != numprocs-1){
	       MPI_Send(cbuf, ib*m, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
	     }
	   }
	 }
       }

       MPI_Barrier(MPI_COMM_WORLD);

       /* output c */
       /* if(myid == 0) */
       /*   printf("ctmp:\n"); */
       /* int id; */
       /* for(ne=0; ne<1; ne++){ */
       /* 	 if(myid == 0) */
       /* 	   printf("ne = %d\n", ne); */
       /* 	 for(id=0; id<numprocs; id++){ */
       /* 	   if(id == myid){ */
       /* 	     for(i=istart; i<iend; i++) */
       /* 	       printf("%f ", ctmp[ne][i]); */
       /* 	   } */
       /* 	   MPI_Barrier(MPI_COMM_WORLD); */
       /* 	 } */
       /* 	 if(myid == 0) */
       /* 	   printf("\n\n"); */
       /* } */
       /* --------------------------------------- */


       /* Backward substitution ------------------ */

       istart = myid * ib;
       iend = (myid + 1) * ib;

       /* xの初期化 */
       for (ne=0; ne<m; ne++)
       	 for (k=0; k<n; k++)
       	   x[ne][k] = 0.0;

       double xbuf[m][ib];
       int ixbuf, jxbuf;

       for (k=(numprocs-1)*ib; k>=0; k-=ib) {
       	 /* 通信用の配列の初期化 */
       	 for(ixbuf=0; ixbuf<m; ixbuf++)
       	   for(jxbuf=0; jxbuf<ib; jxbuf++)
       	     xbuf[ixbuf][jxbuf] = 0;

       	 if (k < iend) {/* 担当するブロックがある */
       	   /* 右隣のPEからデータを受け取る */
       	   if(myid != numprocs-1){
       	     MPI_Recv(xbuf, ib*m, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD, &istatus);
       	   }

       	   idiagPE = k / ib;
       	   if(myid == idiagPE){/* 対角ブロックを持つPE */
       	     for(ne=0; ne<m; ne++){
       	       for(kk=ib-1; kk>=0; kk--){
       		 x[ne][k+kk] = xbuf[ne][kk] + ctmp[ne][k+kk];
       		 for(j=istart+kk+1; j<iend; j++)
       		   x[ne][k+kk] -= A[k+kk][j] * x[ne][j];
       		 x[ne][k+kk] /= A[k+kk][k+kk];
       	       }
       	     }
       	   } else {/* 対角ブロックを持たないPE */
       	     /* 自分の所有範囲のデータのみ計算（まだ最終結果ではない） */
       	     for(ne=0; ne<m; ne++)
       	       for(kk=ib-1; kk>=0; kk--)
       		 for(j=istart; j<iend; j++){
       		   xbuf[ne][kk] -= A[k+kk][j] * x[ne][j];
       		   x[ne][k+kk] = xbuf[ne][kk];
       		 }

       	     /* 左隣のPEに，自分の担当範囲のデータを用いた演算結果を送る */
       	     if(myid != 0){
       	       MPI_Send(xbuf, ib*m, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD);
       	     }
       	   }
       	 }
       }

       MPI_Barrier(MPI_COMM_WORLD);

       /* output x */
       /* if(myid == 0) */
       /*   printf("x:\n"); */
       /* int id; */
       /* for(ne=0; ne<1; ne++){ */
       /* 	 if(myid == 0) */
       /* 	   printf("ne = %d\n", ne); */
       /* 	 for(id=0; id<numprocs; id++){ */
       /* 	   if(id == myid){ */
       /* 	     for(i=istart; i<iend; i++) */
       /* 	       printf("%f ", x[ne][i]); */
       /* 	   } */
       /* 	   MPI_Barrier(MPI_COMM_WORLD); */
       /* 	 } */
       /* 	 if(myid == 0) */
       /* 	   printf("\n\n"); */
       /* } */
       /* --------------------------------------- */
     }


     if(ORIGINAL){
       /* Original */
       /* LU decomposition ---------------------- */
       for (k=0; k<n; k++) {
	 dtemp = 1.0 / A[k][k];
	 for (i=k+1; i<n; i++) {
	   A[i][k] = A[i][k]*dtemp;
	 }
	 for (j=k+1; j<n; j++) {
	   dtemp = A[j][k];
	   for (i=k+1; i<n; i++) {
	     A[j][i] = A[j][i] - A[k][i]*dtemp;
	   }
	 }
       }
       /* --------------------------------------- */


       for (ne=0; ne<m; ne++) {
  
	 /* Forward substitution ------------------ */
	 for (k=0; k<n; k++) {
	   c[k] = b[ne][k];
	   for (j=0; j<k; j++) {
	     c[k] -= A[k][j]*c[j];
	   }
	 }

	 /* if(myid == 0){ */
	 /*   for(k=0; k<n; k++) */
	 /*     printf("%f ", c[k]); */
	 /*   printf("\n\n"); */
	 /* } */
	 /* --------------------------------------- */

	 /* Backward substitution ------------------ */
	 x[ne][n-1] = c[n-1]/A[n-1][n-1];
	 for (k=n-2; k>=0; k--) {
	   x[ne][k] = c[k];
	   for (j=k+1; j<n; j++) {
	     x[ne][k] -= A[k][j]*x[ne][j];
	   }
	   x[ne][k] = x[ne][k] / A[k][k];
	 }
	 /* --------------------------------------- */

       }
       /* End of m loop ----------------------------------------- */
     }

}


