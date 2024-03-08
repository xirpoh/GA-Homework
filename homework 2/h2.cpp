#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <time.h>
#include <iomanip>
#include <fstream>
#include <algorithm>

#define pair_ff pair<float, float>
#define BIT_FLIP(x, p) ((x) ^= (1 << (p)))

#define PI 3.14159
#define PRES 5 
#define DIM 30 
#define POP_SIZE 200
#define MX_SIZE 400
#define MX_GEN 2000
#define CROSSOVR_PROB 0.6
#define MUTATION_PROB 0.005
#define ELITE_SIZE 20
#define LOWER_SIZE 10

#define TESTS 30

using namespace std;
using namespace std::chrono;
long long current_time();
long long start; 
ofstream fout("output_.txt");

random_device rd;
mt19937 engine(rd());
int rand_int(int a, float b);
float rand_float(float a, float b);

pair_ff range;
int l, L;
float range_len, conv_factor;

float (*func)(float*);
float (*func_term)(float, int);

uint32_t v[MX_SIZE][DIM], n;
uint32_t vv[MX_SIZE][DIM];
void rand_bitstring(uint32_t* v);
void mutate_bitstring(uint32_t* v, float);
void print_bitstring(uint32_t* v);
void print_vector(float* x);
void copy_vector(float*, float*);

float x[DIM], saveEval[MX_SIZE];
float testResult[TESTS];
float wheel[MX_SIZE];
pair<float, int> fitness[MX_SIZE];
float funcOffset, f_min, f_max, fit_avg;

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

float eval(uint32_t* v);
void evolve();

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

float eval(uint32_t* v)
{
    for (int i = 0; i < DIM; i++)
        //x[i] = conv_factor * v[i] + range.first;
        x[i] = conv_factor * (v[i] ^ (v[i] >> 1)) + range.first;

    return func(x);
}

void crossover(int a, int b)
{
    int bit = rand_int(0, L - 1);
    int pos = bit / l;

    for (int i = 0; i < pos; i++) {
        v[n    ][i] = v[b][i];
        v[n + 1][i] = v[a][i];
    }

    uint32_t mask = (1 << (l - bit % l)) - 1;
    v[n    ][pos] = (v[b][pos] & ~mask) | (v[a][pos] & mask);
    v[n + 1][pos] = (v[a][pos] & ~mask) | (v[b][pos] & mask);

    for (int i = pos + 1; i < DIM; i++) {
        v[n    ][i] = v[a][i];
        v[n + 1][i] = v[b][i];
    }

    n += 2;
}

int main()
{

    fout << "Function: Schwefel\n\n";
    setup_func(schwefel, schwefel_term, schwefel_range, 418.9829 * DIM);

    //fout << "Function: De Jong\n\n";
    //setup_func(dejong, dejong_term, dejong_range, 0);
    
    //fout << "Function: Rastrigin\n\n";
    //setup_func(rastrigin, rastrigin_term, rastrigin_range, 10 * DIM);

    //fout << "Function: Michalewicz\n\n";
    //setup_func(michalewicz, michalewicz_term, michalewicz_range, 0);
    long long time;

    for (int i = 0; i < TESTS; i++) {
        start = current_time();
        evolve();
        testResult[i] = f_min + funcOffset;
        //fout << testResult[i] << '\n';

        time = (current_time() - start) / 1000;
        cout << "Time: " << time << "s\n";
    }
    fout << '\n';
    fout << "Time: " << time << "s\n";

    return 0;
}

void rand_population()
{
    n = POP_SIZE;
    for (int i = 0; i < POP_SIZE; i++)
        rand_bitstring(v[i]);
}

void mutate(int gen)
{
    for (int i = ELITE_SIZE; i < n; i++)
        mutate_bitstring(v[i], MUTATION_PROB);

    if ((gen + 1) % 100 == 0) {
        for (int i = n - 1; i >= n - LOWER_SIZE; i--)
            mutate_bitstring(v[fitness[i].second], 2 * MUTATION_PROB);
        //    rand_bitstring(v[fitness[i].second]);
    }
}

void evaluate(int gen)
{
    f_min = saveEval[0] = eval(v[0]);
    f_max = saveEval[1] = eval(v[1]);
    float f_avg = f_min;

    for (int i = 2; i < n; i++) {
        saveEval[i] = eval(v[i]);
        f_avg += saveEval[i];

        if (f_min > saveEval[i])
            f_min = saveEval[i];
        else if (f_max < saveEval[i]);
            f_max = saveEval[i];
    }
    
    if ((gen + 1) % 100 == 0)
        cout << f_min + funcOffset << '\n';
    
    float d = f_max - f_min + 0.001;
    float pressure = 4;
    fit_avg = 0;
    for (int i = 0; i < n; i++) {
        fitness[i].first = pow((1 + (f_max - saveEval[i] + funcOffset) / d), pressure);
        //fitness[i].first = pow((1 + (f_max - saveEval[i] + 50) / d), pressure);
        //fitness[i].first = 1 / (saveEval[i] + 10);
        fitness[i].second = i;

        fit_avg += fitness[i].first;
    }

    fit_avg /= n;
}

bool sort_by_fitness(pair<float, int>& a, pair<float, int>& b)
{
    return a.first > b.first;
}

void select()
{
    float total_fitness = 0;
    for (int i = 0; i < n; i++)
        total_fitness += fitness[i].first;

    wheel[0] = 0;
    for (int i = 1; i <= n; i++)
        wheel[i] = fitness[i - 1].first / total_fitness;
    for (int i = 1; i <= n; i++)
        wheel[i] += wheel[i - 1];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < DIM; j++)
            vv[i][j] = v[i][j];
    
    for (int i = ELITE_SIZE; i < POP_SIZE; i++) {
        float r = rand_float(0.00001, 1);
        int j = 0;
        for (; j < n; j++)
            if (wheel[j] < r && r <= wheel[j + 1])
                break;

        for (int k = 0; k < DIM; k++)
            v[i][k] = vv[j][k];

        if (fitness[j].first > fit_avg)
            mutate_bitstring(v[i], MUTATION_PROB);
        else
            mutate_bitstring(v[i], 2 * MUTATION_PROB);
    }

    sort(fitness, fitness + n, sort_by_fitness);
    for (int i = 0; i < ELITE_SIZE; i++)
        for (int k = 0; k < DIM; k++)
            v[i][k] = vv[fitness[i].second][k];

    n = POP_SIZE;
}

void crossover_population()
{
    int parent = -1;
    for (int i = 0; i < POP_SIZE; i++) {
        if (parent < 0) {
            if (rand_float(0, 1) < CROSSOVR_PROB) parent = i;
        }
        else  {
            if (rand_float(0, 1) < CROSSOVR_PROB) {
                crossover(parent, i);
                parent = -1;
            }
        }
    }
}

void evolve()
{
    rand_population();
    evaluate(0);

    for (int g = 0; g < MX_GEN; g++) {
        select();
        //mutate(g);
        crossover_population();
        evaluate(g);
    }
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
    float val = 0;
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

void mutate_bitstring(uint32_t* v, float mut_prob)
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < l; j++)
            if (rand_float(0, 1) < mut_prob)
                BIT_FLIP(v[i], j);
}

void print_bitstring(uint32_t* v)
{
    for (int i = 0; i < DIM; i++) {
        for (int j = l - 1; j >= 0; j--)
            cout << ((v[i] >> j) & 1);
        cout << " ";
    }
    cout << '\n';
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
