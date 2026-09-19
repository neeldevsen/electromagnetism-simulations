#include <iostream>
#include <cmath>
#include <vector>

struct Diagonals
{
    std::vector <double> A_lower {};
    std::vector <double> A_main {};
    std::vector <double> A_upper {};

    explicit Diagonals(int M)
    : A_lower(M-2), A_main(M-1), A_upper(M-2)
    {}
};


auto thomasAlgorithm(const Diagonals& diag, const std::vector<double>& b, const int M)
{
    std::vector<double> rho_p (M-2);
    std::vector<double> mu_p (M-1);

    rho_p[0] = diag.A_upper[0] / diag.A_main[0];
    mu_p[0] = b[0] / diag.A_main[0];

    for (int i {1}; i < M - 2; ++i)
    {
        mu_p[i] = (b[i] / diag.A_main[i] - diag.A_lower[i-1] / diag.A_main[i] * mu_p[i-1]) / (1.0 - diag.A_lower[i-1] / diag.A_main[i] * rho_p[i-1]);
        rho_p[i] = diag.A_upper[i] / diag.A_main[i] / (1.0 - diag.A_lower[i-1] / diag.A_main[i] * rho_p[i-1]);
    }
    mu_p[M-2] = (b[M-2] / diag.A_main[M-2] - diag.A_lower[M-3] / diag.A_main[M-2] * mu_p[M-3]) / (1.0 - diag.A_lower[M-3] / diag.A_main[M-2] * rho_p[M-3]);

    std::vector<double> A (M);
    A[M-2] = mu_p[M-2];
    for (int i{M-3}; i >= 1; --i)
    {
        A[i] = mu_p[i-1] - rho_p[i-1] * A[i+1];
    }
    return A;
}

std::vector<double> generate_b(int M, double L, double J0)
{
    double h {L / (M-1)};
    const double coef {-4 * M_PI * 1e-7 * J0 * h * h };
    std::vector<double> b (M);

    for (int i {0}; i < M; ++i)
    {
        b[i] = coef;
    }
    return b;
}

std::vector<double> generate_B(int M, std::vector<double>& A, double L)
{
    double h {L / (M-1)};
    std::vector<double> B (M-1);
    for (int i{0}; i < M-2; ++i)
    {
        B[i] = -(A[i+2] - A[i]) / (2 * h);
    }
    return B;
}

std::vector<double> generate_b_normally(int M, double L, double J0, double sigma)
{
    double h {L / (M-1)};
    std::vector<double> b (M);

    for (int i {0}; i < M; ++i)
    {
        b[i] = -4 * M_PI * 1e-7 * J0 * h * h * std::exp(-std::pow(h * i -L/2, 2.0) / (2 * sigma * sigma));
    }
    return b;
}


std::vector<double> generate_x(int M, double L, double J0)
{
    double h {L / (M-1)};
    std::vector<double> x (M);

    for (int i {0}; i < M; ++i)
    {
       x[i] = h * i;
    }
    return x;
}



int main()
{
    int M = 1000;
    double L = 1.0;
    double J0 = 1.0;
    double sigma = 0.1;
    std::vector<double> b {generate_b_normally(M, L, J0, sigma)};
    std::vector<double> x {generate_x(M, L, J0)};
    x.pop_back();
    x.pop_back();
    Diagonals diag {M};

    for (int i{0}; i < M-2; ++i)
    {
        diag.A_lower[i] = 1;
        diag.A_main[i] = -2;
        diag.A_upper[i] = 1;
    }
    
    diag.A_main[M-2] = -2;

    std::vector<double> A {thomasAlgorithm(diag, b, M)};

    std::cout << A[0] << '\n';
    std::cout << A[int(M/2)] << '\n';
    std::cout << A[M-1] << '\n';

    std::vector<double> B {generate_B(M, A, L)};

    return 0;
}

