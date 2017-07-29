int myid, numprocs;

ierr = MPI_Init(&argc, &argv); // MPIの初期化
ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid); // ランクの取得
ierr = MPI_Comm_size(MPI_COMM_WORLD, &numproc); // 全プロセス数の取得

ierr = MPI_Barrier(MPI_COMM_WORLD);

ierr = MPI_Reduce();

ierr = MPI_Finalize(); // MPIの終了
