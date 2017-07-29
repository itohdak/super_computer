#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#define  N        576
#define  NPROCS   288

#define  MATRIX  1

#define  EPS    2.220446e-16

double  A_all[N][N];
double  A[N][N/NPROCS];
double  b[N];
double  x[N];
double  c[N];

int     myid, numprocs;

void MyLUsolve(double A[N][N/NPROCS], double b[N], double x[N], int n);

int main(int argc, char* argv[]) {

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
     if (MATRIX == 1) {
       for(j=0; j<N; j++) {
         ii = 0;
         for(i=j; i<N; i++) {
           A_all[j][i] = (N-j) - ii;
           A_all[i][j] = A_all[j][i];
           ii++;
         }
       }

       for(j=0; j<N; j++){
	 ii = 0;
	 for(i=myid*(N/NPROCS); i<(myid+1)*(N/NPROCS); i++){
	   A[j][ii] = A_all[j][i];
	   ii++;
	 }
       }

     } else {
       srand(1);
       dc_inv = 1.0/(double)RAND_MAX;
       for(j=0; j<N; j++) {
         for(i=0; i<N; i++) {
           A_all[j][i] = rand()*dc_inv;
         }
       }
     } /* end of matrix generation -------------------------- */

     /* set vector b  -------------------------- */
     for (i=0; i<N; i++) {
       b[i] = 0.0;
       for (j=0; j<N; j++) {
         b[i] += A_all[i][j];
       }
     }
     /* ----------------------------------------------------- */

     /*
     if (myid == 0) {
      for(j=0; j<N; j++) {
         for(i=0; i<N; i++) {
	   printf("%lf ",A[j][i]);
         }
         printf("\n");
       }
     }
     exit(0);
     */

     /* Start of LU routine ----------------------------*/
     ierr = MPI_Barrier(MPI_COMM_WORLD);
     t1 = MPI_Wtime();

     MyLUsolve(A, b, x, N);

     //ierr = MPI_Barrier(MPI_COMM_WORLD);
     t2 = MPI_Wtime();
     t0 =  t2 - t1; 
     ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     /* End of LU routine --------------------------- */

     if (myid == 0) {
       printf("N  = %d \n",N);
       printf("LU solve time  = %lf [sec.] \n",t_w);

       d_mflops = 2.0/3.0*(double)N*(double)N*(double)N;
       d_mflops += 7.0/2.0*(double)N*(double)N;
       d_mflops += 4.0/3.0*(double)N;
       d_mflops = d_mflops/t_w;
       d_mflops = d_mflops * 1.0e-6;
       printf(" %lf [MFLOPS] \n", d_mflops);
     }

     /* Verification routine ----------------- */
     ib = N / NPROCS;
     dtemp_t = 0.0;
     for(j=myid*ib; j<(myid+1)*ib; j++) {
       dtemp2 = x[j] - 1.0;
       dtemp_t += dtemp2*dtemp2;
     }
     dtemp_t = sqrt(dtemp_t);
     /* -------------------------------------- */

     MPI_Reduce(&dtemp_t, &dtemp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

     /* Do not modify follows. -------- */ 
     if (myid == 0) {
       if (MATRIX == 1) dtemp2 = (double)N*(double)N*(double)N;
       else dtemp2 = (double)N*(double)N;
       dtemp_t = EPS*(double)N*dtemp2;
       printf("Pass value: %e \n", dtemp_t);
       printf("Calculated value: %e \n", dtemp);
       if (dtemp > dtemp_t) {
          printf("Error! Test is falled. \n");
          exit(1);
       } 
       printf(" OK! Test is passed. \n");
     }
     /* ----------------------------------------- */

     ierr = MPI_Finalize();

     exit(0);
}

void MyLUsolve(double A[N][N/NPROCS], double b[N], double x[N], int n) 
{
     int i, j, k;
     double dtemp, dtemp2;
     int istart, iend;
     int start, end;
     int ib = n / numprocs;
     MPI_Status istatus;
     int idiagPE;

     /* LU decomposition ---------------------- */
     istart = myid * ib;
     iend = (myid + 1) * ib;
     start = myid * ib;
     end = (myid + 1) * ib;
     int isendPE;

     for (k=0; k<iend; k++) {
       double buf[n-k-1]; // 通信用のバッファ
       idiagPE = k / ib;
       if (myid == idiagPE){// 枢軸ベクトルを持つPE
     	 dtemp = 1.0 / A[k][k-start];
     	 for (i=k+1; i<n; i++) {
     	   dtemp2 = A[i][k-start]*dtemp;
     	   A[i][k-start] = dtemp2;
     	   buf[i-k-1] = dtemp2;
     	 }
	 /* for (isendPE=myid+1; isendPE<numprocs; isendPE++){ */
	 /*   MPI_Send(buf, n-k-1, MPI_DOUBLE, isendPE, k, MPI_COMM_WORLD); */
	 /* } */
     	 if(myid != numprocs-1)
     	   MPI_Send(buf, n-k-1, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
     	 istart = k+1;
       } else {// 枢軸ベクトルを持たないPE
     	 /* MPI_Recv(buf, n-k-1, MPI_DOUBLE, idiagPE, k, MPI_COMM_WORLD, &istatus); */
     	 MPI_Recv(buf, n-k-1, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD, &istatus);
     	 if(myid != numprocs-1)
     	   MPI_Send(buf, n-k-1, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
       }

       for (i=k+1; i<n; i++) {
     	 dtemp = buf[i-k-1];
     	 /* A[i][k-start] = dtemp; */
     	 for (j=istart; j<iend; j++) {
     	   A[i][j-start] = A[i][j-start] - A[k][j-start]*dtemp;
     	 }
       }
     }

     MPI_Barrier(MPI_COMM_WORLD);

     /* int id; */
     /* for(id=0; id<NPROCS; id++){ */
     /*   if(myid == id){ */
     /* 	 for(j=0; j<N/NPROCS; j++) */
     /* 	   printf("%f ", A[n-3][j]); */
     /*   } */
     /*   MPI_Barrier(MPI_COMM_WORLD); */
     /* } */
     /* if(myid == 0) */
     /*   printf("\n"); */
     /* --------------------------------------- */


     /* Forward substitution ------------------ */
     istart = myid * ib;
     iend = (myid + 1) * ib;

     int kk;

     /* cの初期化 */
     for (k=0; k<n; k++)
       c[k] = 0.0;
     
     for (k=0; k<n; k+=ib) {
       if (k >= istart) {/* 担当するブロックがある */
     	 /* 左隣のPEからデータを受け取る */
     	 if(myid != 0){
     	   MPI_Recv(&c[k], ib, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD, &istatus);
     	 }

     	 idiagPE = k / ib;
     	 if(myid == idiagPE){/* 対角ブロックを持つPE */
     	   for(kk=0; kk<ib; kk++){
     	     c[k+kk] += b[k+kk];
     	     for(j=istart; j<istart+kk; j++){
     	       c[k+kk] -= A[k+kk][j-start] * c[j];
	     }
     	   }
     	 } else {/* 対角ブロックを持たないPE */
     	   /* 自分の所有範囲のデータのみ計算（まだ最終結果ではない） */
     	   for(kk=0; kk<ib; kk++)
     	     for(j=istart; j<iend; j++){
	       c[k+kk] -= A[k+kk][j-start] * c[j];
	     }

     	   /* 右隣のPEに，自分の担当範囲のデータを用いた演算結果を送る */
     	   if(myid != numprocs-1){
     	     MPI_Send(&c[k], ib, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD);
     	   }
     	 }
       }
     }

     MPI_Barrier(MPI_COMM_WORLD);

     /* output c */
     /* if(myid == 0) */
     /*   printf("c:\n"); */
     /* int id; */
     /* for(id=0; id<numprocs; id++){ */
     /*   if(id == myid) */
     /* 	 for(i=istart; i<iend; i++) */
     /* 	   printf("%f ", c[i]); */
     /*   MPI_Barrier(MPI_COMM_WORLD); */
     /* } */
     /* if(myid == 0) */
     /*   printf("\n\n"); */
     /* --------------------------------------- */


     /* Backward substitution ------------------ */
     istart = myid * ib;
     iend = (myid + 1) * ib;

     /* xの初期化 */
     for (k=0; k<n; k++)
       x[k] = 0.0;

     for (k=(numprocs-1)*ib; k>=0; k-=ib) {
       if (k < iend) {/* 担当するブロックがある */
     	 /* 右隣のPEからデータを受け取る */
     	 if(myid != numprocs-1){
     	   MPI_Recv(&x[k], ib, MPI_DOUBLE, myid+1, k, MPI_COMM_WORLD, &istatus);
     	 }

     	 idiagPE = k / ib;
     	 if(myid == idiagPE){/* 対角ブロックを持つPE */
     	   for(kk=ib-1; kk>=0; kk--){
     	     x[k+kk] += c[k+kk];
     	     for(j=istart+kk+1; j<iend; j++)
     	       x[k+kk] -= A[k+kk][j-start] * x[j];
     	     x[k+kk] /= A[k+kk][k+kk-start];
     	   }
     	 } else {/* 対角ブロックを持たないPE */
     	   /* 自分の所有範囲のデータのみ計算（まだ最終結果ではない） */
     	   for(kk=ib-1; kk>=0; kk--)
     	     for(j=istart; j<iend; j++)
     	       x[k+kk] -= A[k+kk][j-start] * x[j];

     	   /* 左隣のPEに，自分の担当範囲のデータを用いた演算結果を送る */
     	   if(myid != 0){
     	     MPI_Send(&x[k], ib, MPI_DOUBLE, myid-1, k, MPI_COMM_WORLD);
     	   }
     	 }
       }
     }

     MPI_Barrier(MPI_COMM_WORLD);

     /* output x */
     /* if(myid == 0) */
     /*   printf("x:\n"); */
     /* /\* int id; *\/ */
     /* for(id=0; id<numprocs; id++){ */
     /*   if(id == myid) */
     /* 	 for(i=istart; i<iend; i++) */
     /* 	   printf("%f ", x[i]); */
     /*   MPI_Barrier(MPI_COMM_WORLD); */
     /* } */
     /* if(myid == 0) */
     /*   printf("\n\n"); */
     /* --------------------------------------- */


     /* Original */
     /* /\* LU decomposition ---------------------- *\/ */
     /* for (k=0; k<n; k++) { */
     /*   dtemp = 1.0 / A[k][k]; */
     /*   for (i=k+1; i<n; i++) { */
     /*     A[i][k] = A[i][k]*dtemp; */
     /*   } */
     /*   for (j=k+1; j<n; j++) { */
     /*     dtemp = A[j][k]; */
     /*     for (i=k+1; i<n; i++) { */
     /*       A[j][i] = A[j][i] - A[k][i]*dtemp; */
     /*     } */
     /*   } */
     /* } */

     /* /\* output A *\/ */
     /* if(myid == 0){ */
     /*   printf("A:\n"); */
     /*   for(i=0; i<n; i++){ */
     /* 	 for(j=0; j<n; j++){ */
     /* 	   printf("%f ", A[i][j]); */
     /* 	 } */
     /* 	 printf("\n\n"); */
     /*   } */
     /* } */
     /* /\* --------------------------------------- *\/ */


     /* /\* Forward substitution ------------------ *\/ */
     /* for (k=0; k<n; k++) { */
     /*   c[k] = b[k]; */
     /*   for (j=0; j<k; j++) { */
     /*     c[k] -= A[k][j]*c[j]; */
     /*   } */
     /* } */

     /* /\* output c *\/ */
     /* if(myid == 0){ */
     /*   printf("c:\n"); */
     /*   for(i=0; i<n; i++) */
     /* 	 printf("%f ", c[i]); */
     /*   printf("\n\n"); */
     /* } */
     /* /\* --------------------------------------- *\/ */


     /* /\* Backward substitution ------------------ *\/ */
     /* x[n-1] = c[n-1]/A[n-1][n-1]; */
     /* for (k=n-2; k>=0; k--) { */
     /*   x[k] = c[k]; */
     /*   for (j=k+1; j<n; j++) { */
     /*     x[k] -= A[k][j]*x[j]; */
     /*   } */
     /*   x[k] = x[k] / A[k][k]; */
     /* } */

     /* /\* output x *\/ */
     /* if(myid == 0){ */
     /*   printf("x:\n"); */
     /*   for(i=0; i<n; i++) */
     /* 	 printf("%f ", x[i]); */
     /*   printf("\n\n"); */
     /* } */
     /* /\* --------------------------------------- *\/ */

}

