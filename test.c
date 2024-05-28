#include <petscdmda.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>

int main(int argc, char **args) {
  PetscInitialize(&argc, &args, (char *)0, "Solve a linear system with DMDA");
  Vec x, b; /* 近似解和右手边向量 */
  Mat A;    /* 系数矩阵 */
  KSP ksp;  /* 线性求解器上下文 */
  DM da;    /* 分布式数组上下文 */
  PetscErrorCode ierr;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz,
      M = 10, N = 10, P = 10; /* 全局网格大小 */
  MatStencil row[2], col[2];
  PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL);
  N = M;
  P = M;
  PetscScalar v;
  PetscScalar val_A[2][2];



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
  PetscCall(
      DMDAGetCorners(da, &startx, &starty, &startz, &nx, &ny, &nz));

  /* 设置矩阵A */
  for (ez = startz; ez < startz + nz; ez++) {
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        if (ex >= 1) {
          row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = 1;
          val_A[0][1] = -1;
          val_A[1][0] = -1;
          val_A[1][1] = 1;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ey >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = 1;
          val_A[0][1] = -1;
          val_A[1][0] = -1;
          val_A[1][1] = 1;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ez >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = 1;
          val_A[0][1] = -1;
          val_A[1][0] = -1;
          val_A[1][1] = 1;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Initialized.\n"));
  /* 求解线性系统 */
  ierr = KSPSolve(ksp, b, x);
  CHKERRQ(ierr);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Done.\n"));
  /* 打印解向量 */

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
