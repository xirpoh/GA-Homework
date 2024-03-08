#include <iostream>
#include <fstream>

using namespace std;
ofstream fout("h1p_FI.txt");

int f(int x)
{
    return x * x * x - 60 * x * x + 900 * x + 100;
}

void BI(int x)
{
    int mx = f(x), b = x;
    for (int i = 4; i >= 0; i--) {
        int n = x ^ (1 << i);
        if (f(n) > mx)
            mx = f(n), b = n;

        fout << n << "(" << f(n) << ") ";
    }

    if (b != x)
        fout << "-> " << b << "\n";
    else
        fout << "local\n";

    fout << "\n\n";
}

void FI(int x)
{
    int mx = f(x), b = x;
    for (int i = 4; i >= 0; i--) {
        int n = x ^ (1 << i);
        if (f(n) > mx) {
            mx = f(n), b = n;
            break;
        }
    }
    if (b != x)
        fout << "-> " << b << "\n";
    else
        fout << "local\n";
}

int main()
{
    for (int i = 0; i < 32; i++) {
        fout << i << "(" << f(i) << ") ";
        FI(i);
    }
}
