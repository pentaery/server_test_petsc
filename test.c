#include <petscdmda.h>
#include <petscksp.h>
#include <petscoptions.h>
#include <petscsys.h>

int main(int argc, char **args) {
  Vec x, b; /* 近似解和右手边向量 */
  Mat A;    /* 系数矩阵 */
  KSP ksp;  /* 线性求解器上下文 */
  DM da;    /* 分布式数组上下文 */
  PetscErrorCode ierr;
  PetscInt i, j, k, M = 10, N = 10, P = 10; /* 全局网格大小 */
  PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL);
  N = M;
  P = M;
  PetscMPIInt rank, size;
  PetscScalar v;

  PetscInitialize(&argc, &args, (char *)0, "Solve a linear system with DMDA");

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
  CHKERRQ(ierr);

  /* 创建一个三维DMDA */
  ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                      DM_BOUNDARY_NONE,  /* 边界条件 */
                      DMDA_STENCIL_STAR, /* 星型模板 */
                      M, N, P,           /* 全局尺寸 */
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, /* 进程划分 */
                      1,                                        /* 自由度 */
                      1,                /* 额外条目 */
                      NULL, NULL, NULL, /* 分布向量 */
                      &da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);
  CHKERRQ(ierr);
  ierr = DMSetUp(da);
  CHKERRQ(ierr);

  /* 创建矩阵和向量 */
  ierr = DMCreateMatrix(da, &A);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &b);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &x);
  CHKERRQ(ierr);

  /* 设置右手边向量b */
  ierr = VecSet(b, 1.0);
  CHKERRQ(ierr);

  /* 获取本地范围 */
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da, &info);
  CHKERRQ(ierr);

  /* 设置矩阵A */
  for (i = info.xs; i < info.xs + info.xm; i++) {
    for (j = info.ys; j < info.ys + info.ym; j++) {
      for (k = info.zs; k < info.zs + info.zm; k++) {
        if (i > 0) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             ((i - 1) * N * P + j * P + k), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        if (i < M - 1) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             ((i + 1) * N * P + j * P + k), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        if (j > 0) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             (i * N * P + (j - 1) * P + k), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        if (j < N - 1) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             (i * N * P + (j + 1) * P + k), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        if (k > 0) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             (i * N * P + j * P + (k - 1)), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        if (k < P - 1) {
          v = -1.0;
          ierr = MatSetValue(A, (i * N * P + j * P + k),
                             (i * N * P + j * P + (k + 1)), v, INSERT_VALUES);
          CHKERRQ(ierr);
        }
        v = 6.0;
        ierr = MatSetValue(A, (i * N * P + j * P + k), (i * N * P + j * P + k),
                           v, INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }

  /* 完成矩阵装配 */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  /* 创建线性求解器上下文 */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);

  /* 求解线性系统 */
  ierr = KSPSolve(ksp, b, x);
  CHKERRQ(ierr);

  /* 打印解向量 */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "solved\n"));
  CHKERRQ(ierr);

  /* 释放内存 */
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = VecDestroy(&x);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = MatDestroy(&A);
  CHKERRQ(ierr);
  ierr = DMDestroy(&da);
  CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
