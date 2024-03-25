#include <iostream>
#include "Eigen\Eigen"

using namespace Eigen;
using namespace std;

bool SolveSystem(const Matrix2d& A,
                 const Vector2d& b,
                 double& detA,
                 double& condA,
                 double& errRelPALU,
                 double& errRelQR)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();
    detA = A.determinant();

    if (singularValuesA.minCoeff() < 1e-16)
    {
        errRelPALU = -1;
        errRelQR = -1;
        return false;
    }
    Vector2d exactSol;
    exactSol << -1.00e+0, -1.00e+0;

    Vector2d xPALU = A.fullPivLu().solve(b); // PALU solution

    errRelPALU = (exactSol - xPALU).norm()/exactSol.norm();

    Vector2d xQR = A.fullPivHouseholderQr().solve(b); //QR solution

    errRelQR = (exactSol - xQR).norm()/exactSol.norm();
    return true;
}

int main()
{
    Matrix2d A1, A2, A3;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b1, b2, b3;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    double detA1, condA1, errRelA1_PALU, errRelA1_QR;
    if(SolveSystem(A1, b1, detA1, condA1, errRelA1_PALU, errRelA1_QR))
        cout <<scientific<< "A1 - det: "<<detA1<<", cond: "<<1.0/condA1<<", relative error with PALU decomposition: "<<errRelA1_PALU<<", relative error with QR decomposition: "<< errRelA1_QR << endl;
    else
        cout <<scientific<<"A1 - det: "<<detA1<< ", cond: "<< 1.0/condA1<< "Matrix is singular"<<endl ;

    double detA2, condA2, errRelA2_PALU, errRelA2_QR;
    if(SolveSystem(A2, b2, detA2, condA2, errRelA2_PALU, errRelA2_QR))
        cout <<scientific<< "A2 - det: "<<detA2<<", cond: "<<1.0/condA2<<", relative error with PALU decomposition: "<<errRelA2_PALU<<", relative error with QR decomposition: "<< errRelA2_QR << endl;
    else
        cout <<scientific<<"A2 - det: "<<detA2<< ", cond: "<< 1.0/condA2<< "Matrix is singular"<<endl ;

    double detA3, condA3, errRelA3_PALU, errRelA3_QR;
    if(SolveSystem(A3, b3, detA3, condA3, errRelA3_PALU, errRelA3_QR))
        cout <<scientific<< "A3 - det: "<<detA3<<", cond: "<<1.0/condA3<<", relative error with PALU decomposition: "<<errRelA3_PALU<<", relative error with QR decomposition: "<< errRelA3_QR << endl;
    else
        cout <<scientific<<"A3 - det: "<<detA3<< ", cond: "<< 1.0/condA3<< "Matrix is singular"<<endl ;


    return 0;
}
