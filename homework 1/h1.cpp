#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <time.h>
#include <iomanip>
#include <bitset>
#include <fstream>

#define pair_ff pair<float, float>
#define BIT_FLIP(x, p) ((x) ^= (1 << (p)))

#define PI 3.14159
#define PRES 5 
#define DIM 10 

#define TESTS 1
#define HC_BI 0 
#define HC_FI 1
#define HC_WI 2
#define HC_ITER 10000 
#define SA_ITER 500

using namespace std;
using namespace std::chrono;
long long current_time();
long long start; 

random_device rd;
mt19937 engine(rd());
int rand_int(int a, float b);
float rand_float(float a, float b);

pair_ff range;
int l, L;
float range_len, conv_factor;

float (*func)(float*);
float (*func_term)(float, int);
float update_funcExcept(float*);

uint32_t v[DIM];
void rand_bitstring(uint32_t* v);
void print_vector(float* x);
void copy_vector(float*, float*);

float x[DIM], x_min[DIM], funcExcept[DIM], saveTerm[DIM];
float funcOffset;
float testResult[TESTS];

pair_ff dejong_range(-5.12, 5.12);
float dejong(float* x);
float dejong_term(float, int);

pair_ff schwefel_range(-500, 500);
float schwefel(float* x);
float schwefel_term(float, int);

pair_ff rastrigin_range(-5.12, 5.12);
float rastrigin(float* x);
float rastrigin_term(float, int);

pair_ff michalewicz_range(0, PI);
float michalewicz(float* x);
float michalewicz_term(float, int);

float eval(uint32_t* v, bool neigh = 0, int pos = 0);
ofstream fout("output_.txt");

void setup_func(float (*f)(float*), float (*f_term)(float, int), pair_ff r, float o)
{
    funcOffset = o;
    func = f;
    func_term = f_term;
    range = r;
    
    range_len = range.second - range.first;
    l = ceil(log2(pow(10, PRES) * range_len));
    L = l * DIM;

    conv_factor = range_len / ((1 << l) - 1);
}

void interpret_results()
{
    float f_min = testResult[0], f_max = f_min;
    float median = 0, st_dev = 0;

    for (int i = 0; i < TESTS; i++) {
        if (testResult[i] < f_min) f_min = testResult[i];
        if (testResult[i] > f_max) f_max = testResult[i];

        median += testResult[i];
    }
    median /= TESTS;

    for (int i = 0; i < TESTS; i++)
        st_dev += pow((testResult[i] - median), 2); 
    st_dev = sqrt(st_dev / TESTS);

    fout << f_min << '\n' << f_max << '\n' << median << '\n' << st_dev << "\n\n"; 
}

bool improve_neighborhood(uint32_t* v, float& curr, int variant)
{
    int best_bit, best_pos;
    float fitness, best = curr;
    bool halt = 0, accept, improved = 0;

    for (int i = 0; !halt && i < DIM; i++) {
        for (int j = 0; !halt && j < l; j++) {
            
            BIT_FLIP(v[i], j);
            fitness = eval(v, 1, i);
            accept = 0;

            if (variant == HC_WI) {
                if (!improved && fitness < best)
                    improved = 1, accept = 1;
                else if (improved && fitness < curr && fitness > best)
                    accept = 1;
            }
            else if (fitness < best) {
                if (variant == HC_FI)
                    halt = 1;
                accept = 1;
            }

            if (accept) {
                best = fitness;
                best_bit = j;
                best_pos = i;
            }

            BIT_FLIP(v[i], j);
        }
    }
    
    if (best < curr) {
        curr = best;
        BIT_FLIP(v[best_pos], best_bit);

        x[best_pos] = conv_factor * v[best_pos] + range.first;
        update_funcExcept(x);

        return true;
    }

    return false;
}


void Hill_Climbing(int variant)
{
    start = current_time();

    for (int i = 0; i < TESTS; i++) {
        int iterations = HC_ITER;

        rand_bitstring(v);
        float best = eval(v), curr_fitness;

        while (iterations--) {
            rand_bitstring(v);
            curr_fitness = eval(v);
            
            while (improve_neighborhood(v, curr_fitness, variant));

            if (curr_fitness < best) {
                best = curr_fitness;
                //copy_vector(x_min, x);
            }
        }

        testResult[i] = best + funcOffset;
        cout << testResult[i] << '\n';
        //print_vector(x_min);
    }

    long long time = (current_time() - start) / 1000;
    fout << "Time: " << time << "s\n";
    cout << "Time: " << time << "s\n";
    interpret_results();
}

void Simulated_Annealing(int test)
{
    int runs = 3, iterations = SA_ITER;
    int bit, pos;
    rand_bitstring(v);
    float curr_fitness = eval(v), best = curr_fitness, fitness;
    
    while (runs--) {
        rand_bitstring(v);
        curr_fitness = eval(v);
        
        for (float T = 100; T > 10e-8; T *= 0.99) {
            for (int j = 0; j < iterations; j++) {
                bit = rand_int(0, L - 1);
                pos = bit / l;

                BIT_FLIP(v[pos], bit % l); 
                fitness = eval(v, 1, pos);
            
                bool accept = 0;
                if (fitness < curr_fitness)
                    accept = 1;
                else if (rand_float(0, 0.99999) < exp(-abs(fitness - curr_fitness) / T))
                    accept = 1;

                if (accept) {
                    curr_fitness = fitness;
                    x[pos] = conv_factor * v[pos] + range.first;
                    update_funcExcept(x);
                }
                else
                    BIT_FLIP(v[pos], bit % l); 
            }
        }

        if (curr_fitness < best)
            best = curr_fitness;
    }
    
    testResult[test] = best + funcOffset;
    cout << testResult[test] << '\n';
}

void run_Simulated_Annealing()
{
    start = current_time();

    for (int i = 0; i < TESTS; i++)
        Simulated_Annealing(i);

    long long time = (current_time() - start) / 1000;
    fout << "Time: " << time << "s\n";
    cout << "Time: " << time << "s\n";
    interpret_results();
}

float eval(uint32_t* v, bool neigh, int pos)
{
    if (neigh) {
        float t = conv_factor * v[pos] + range.first;
        return funcExcept[pos] + func_term(t, pos);
    }

    for (int i = 0; i < DIM; i++)
        x[i] = conv_factor * v[i] + range.first;

    return update_funcExcept(x);
}

void run_tests()
{
    fout << "HC_BI\n";
    Hill_Climbing(HC_BI);

    fout << "HC_FI\n";
    Hill_Climbing(HC_FI);

    fout << "HC_WI\n"; 
    Hill_Climbing(HC_WI);

    fout << "SA\n";
    run_Simulated_Annealing();
}

int main()
{
    fout << "Tests: " << TESTS << '\n';
    fout << "Dimensions: " << DIM << '\n';
    fout << "HC_ITER: " << HC_ITER << '\n';
    fout << "SA_ITER: " << SA_ITER << '\n';

    fout << "Function: Schwefel\n\n";
    setup_func(schwefel, schwefel_term, schwefel_range, 418.9829 * DIM);
    run_tests();

    /*fout << "Function: De Jong\n\n";
    setup_func(dejong, dejong_term, dejong_range, 0);
    run_tests();
    
    fout << "Function: Rastrigin\n\n";
    setup_func(rastrigin, rastrigin_term, rastrigin_range, 10 * DIM);
    run_tests();

    fout << "Function: Michalewicz\n\n";
    setup_func(michalewicz, michalewicz_term, michalewicz_range, 0);
    run_tests();*/
    
    return 0;
}

float update_funcExcept(float* x)
{
    funcExcept[0] = 0;
    for (int i = 1; i < DIM; i++) {
        saveTerm[i - 1] = func_term(x[i - 1], i - 1);
        funcExcept[i] = saveTerm[i - 1] + funcExcept[i - 1];
    }
    saveTerm[DIM - 1] = func_term(x[DIM - 1], DIM - 1);

    float suffix = 0;
    for (int i = DIM - 1; i >= 0; i--) {
        funcExcept[i] += suffix;
        suffix += saveTerm[i];
    }
    
    return suffix;
}

float dejong_term(float t, int i)
{
    return t * t;
}

float dejong(float* x)
{
    float val = 0;
    for (int i = 0; i < DIM; i++)
        val += (x[i] * x[i]);

    return val;
}

float schwefel_term(float t, int i)
{
    return -(t * sin(sqrt(abs(t))));
}

float schwefel(float* x)
{
    float val = 0;
    for (int i = 0; i < DIM; i++)
        val -= (x[i] * sin(sqrt(abs(x[i]))));

    return val;
}

float rastrigin_term(float t, int i)
{
    return (t * t - 10 * cos(2.0 * PI * t));
}

float rastrigin(float* x)
{
    float val = 10 * DIM;
    for (int i = 0; i < DIM; i++) {
        val += (x[i] * x[i] - 10 * cos(2.0 * PI * x[i]));
    }
    return val;
}

float michalewicz_term(float t, int i)
{
    float arg = ((i + 1) * (t * t)) / PI;
    return -(sin(t) * pow(sin(arg), 20));
}

float michalewicz(float* x)
{
    float val = 0;
    float two_m = 20;

    for (int i = 0; i < DIM; i++) {
        float arg = ((i + 1) * (x[i] * x[i])) / PI;
        val -= sin(x[i]) * pow(sin(arg), two_m);
    }

    return val;
}

int rand_int(int a, float b)
{
    uniform_int_distribution<int> dist(a, b);
    return dist(engine);
}

float rand_float(float a, float b)
{
    uniform_real_distribution<float> dist(a, b);
    return dist(engine);
}

void rand_bitstring(uint32_t* v)
{
    for (int i = 0; i < DIM; i++)
        v[i] = rand_int(0, (1 << l) - 1);
}

void copy_vector(float* a, float* b)
{
    for (int i = 0; i < DIM; i++)
        a[i] = b[i];
}

long long current_time() 
{
    milliseconds ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    );

    return ms.count();
}

void print_vector(float* x)
{
    for (int i = 0; i < DIM; i++) 
        cout << x[i] << " ";
    cout << '\n';
}
