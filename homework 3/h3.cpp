#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <time.h>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <string>

#define pair_ff pair<float, float>
#define BIT_FLIP(x, p) ((x) ^= (1 << (p)))

#define PI 3.14159
#define PRES 5 
#define MX_N 6000
#define POP_SIZE 200
#define MX_SIZE 1000
#define MX_GEN 2000
#define CROSSOVR_PROB 0.9
#define MUTATION_PROB   0.001
#define MUTATION_PROB_2 0.05
#define ELITE_SIZE 20
#define LOWER_SIZE 50

#define TESTS 1

using namespace std;
using namespace std::chrono;
long long current_time();
long long start; 
ofstream fout("output_.txt");

random_device rd;
mt19937 engine(rd());
int rand_int(int a, float b);
float rand_float(float a, float b);

uint32_t N;
uint32_t v[MX_SIZE][MX_N], n, saveEval[MX_SIZE];
uint32_t vv[MX_SIZE][MX_N];
uint32_t p[MX_N], cycle[MX_N];
float cost[MX_N][MX_N];
void rand_chromosome(uint32_t* v); 
void mutate_chromosome(uint32_t* v, float mut_prob); 

int testResult[TESTS];
float wheel[MX_SIZE];
pair<float, int> fitness[MX_SIZE];
float funcOffset, f_min, f_max, fit_avg, mut_prob, mut_prob2;

int eval(uint32_t* v);
void evolve();
void gen_greedy_instance(int start);
void encode_chromosome(uint32_t* v);
void decode_chromosome(uint32_t* v);

void setup_instance(string file)
{
    ifstream fin(file);
    string line; getline(fin, line);
    cout << line << '\n';
    fout << line << '\n';
    fin >> N;
    //mut_prob = MUTATION_PROB;
    //mut_prob2 = MUTATION_PROB_2;
    mut_prob = 0.7 * int((1.0 / N) * 10000) / 10000.0;
    mut_prob2 = 5 * mut_prob;
    cout << mut_prob << '\n';
    n = POP_SIZE;
    
    pair<float, float> coord[N];
    int node;
    float x, y;
    for (int i = 0; i < N; i++) {
        fin >> node >> x >> y;
        coord[i].first = x;
        coord[i].second = y;
    }

    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++) {
            float dx = coord[i].first - coord[j].first;
            float dy = coord[i].second - coord[j].second;
            cost[i][j] = cost[j][i] = sqrt(dx * dx + dy * dy);
        }

    fin.close();
}

void gen_greedy_instance(uint32_t* v, int start) 
{
    int nod = start;
    v[0] = nod;

    vector<int> id(N);
    for (int i = 0; i < N; i++)
        id[i] = i;
    id[start] = N - 1;

    for (int i = 1; i < N; i++) {
        int cmin = cost[nod][id[0]], next = 0;

        for (int j = 0; j < N - i; j++)
            if (cost[nod][id[j]] < cmin)
                cmin = cost[nod][id[j]], next = j;

        v[i] = nod = id[next];
        id[next] = id[N - i - 1];

        //fout << v[i] << ' ';
    }
    //exit(0);
}

int main()
{
    setup_instance("berlin52.tsp");
    //setup_instance("eil101.tsp");
    //setup_instance("d198.tsp");
    //setup_instance("pr226.tsp");
    //setup_instance("a280.tsp");
    //setup_instance("rat575.tsp");
    //setup_instance("pr1002.tsp");
    //setup_instance("nrw1379.tsp");
    //setup_instance("pr2392.tsp");
    //setup_instance("rl5915.tsp");
    start = current_time();
    long long time;

    for (int i = 0; i < TESTS; i++) {
        start = current_time();
        evolve();
        testResult[i] = f_min;
        //fout << testResult[i] << '\n';

        time = (current_time() - start) / 1000;
        cout << "Time: " << time << "s\n";
    }
    fout << '\n';
    fout << "Time: " << time << "s\n";

    return 0;
}

void crossover(int a, int b)
{
    int l = rand_int(0, N - 1);

    for (int i = 0; i < l; i++) {
        v[n    ][i] = v[a][i];
        v[n + 1][i] = v[b][i];
    }

    for (int i = l; i < N; i++) {
        v[n    ][i] = v[b][i];
        v[n + 1][i] = v[a][i];
    }

    n += 2;
}

bool f_a[MX_N], f_b[MX_N];

void crossover_OX(int a, int b)
{
    int l1, l2;
    do {
        l1 = rand_int(0, N - 1);
        l2 = rand_int(0, N - 1);
    } while (l1 == l2);
    if (l1 > l2) swap(l1, l2);

    for (int i = 0; i < N; i++)
        if (i >= l1 && i <= l2) {
            f_a[v[a][i]] = 1;
            f_b[v[b][i]] = 1;
        }
        else {
            f_a[v[a][i]] = 0;
            f_b[v[b][i]] = 0;
        }

    for (int i = 0, b_idx = 0, a_idx = 0; i < N; i++) {
        if (i >= l1 && i <= l2) {
            v[n    ][i] = v[a][i];
            v[n + 1][i] = v[b][i];
        }
        else {
            while (f_a[v[b][b_idx]]) b_idx++;
            while (f_b[v[a][a_idx]]) a_idx++;
        
            v[n    ][i] = v[b][b_idx++];
            v[n + 1][i] = v[a][a_idx++];
        }
    }

    n += 2;
}

void rand_chromosome(uint32_t* v) 
{
    vector<int> p(N), id(N);
    for (int i = 0; i < N; i++) {
        p[i] = rand_int(0, N - 1 - i);
        id[i] = i;
    }

    for (int i = 0; i < N; i++) {
        v[i] = id[p[i]];
        id[p[i]] = id[N - i - 1];
    }
}

void rand_population()
{
    for (int i = 0; i < POP_SIZE; i++)
        rand_chromosome(v[i]);
}

void encode_chromosome(uint32_t* v)
{
    for (int i = 0; i < N; i++)
        v[i] = cycle[i];
}

void rand_greedy_population()
{
    n = POP_SIZE;
    for (int i = 0; i < POP_SIZE; i++) {
        gen_greedy_instance(v[i], rand_int(0, N - 1));
    }
}

void mutate_chromosome(uint32_t* v, float mut_prob, int way)
{
    for (int i = 0; i < N - 1; i++) {
        if (rand_float(0, 1) < mut_prob2) {
            int r = rand_int(0, N - 1);
            int rr = r + 1;
            if (rr == N) rr = 0;

            if (cost[v[i]][v[i + 1]] + cost[v[r]][v[rr]] > 
                cost[v[i]][v[r]] + cost[v[i + 1]][v[rr]])
                swap(v[i + 1], v[r]);
        }
    }
    
    if (rand_float(0, 1) < mut_prob) {
        int l1, l2;
        do {
            l1 = rand_int(0, N - 1);
            l2 = rand_int(0, N - 1);
        } while (l1 == l2); 
        if (l1 > l2)
            swap(l1, l2);

        reverse(v + l1, v + l2);
    }
}


void mutate(int gen)
{
    for (int i = ELITE_SIZE; i < POP_SIZE; i++) 
        mutate_chromosome(v[i], MUTATION_PROB);
}

void decode_chromosome(uint32_t* v) {
    for (int i = 0; i < N; i++)
        cycle[i] = v[i];

    /*for (int i = 0; i < N; i++)
        p[i] = i;

    for (int i = 0; i < N; i++) {
        cycle[i] = p[v[i]];

        for (int j = v[i] + 1; j < N - i; j++)
            p[j - 1] = p[j];
        //p[v[i]] = p[N - 1 - i];
    }*/

}

int eval(uint32_t* v)
{
    float cycle_len = cost[v[0]][v[N - 1]];
    for (int i = 0; i < N - 1; i++)
        cycle_len += cost[v[i]][v[i + 1]];

    return cycle_len;
}

bool sort_by_fitness(pair<float, int>& a, pair<float, int>& b)
{
    return a.first > b.first;
}

void evaluate(int gen) 
{
    f_min = saveEval[0] = eval(v[0]);
    f_max = saveEval[1] = eval(v[1]);
    float f_avg = f_min;
    int idx = 0;

    for (int i = 2; i < n; i++) {
        saveEval[i] = eval(v[i]);
        f_avg += saveEval[i];

        if (f_min > saveEval[i])
            f_min = saveEval[i], idx = i;
        else if (f_max < saveEval[i]);
            f_max = saveEval[i];
    }

    if ((gen + 1) % 500 == 0) {
        cout << f_min << '\n';
        //fout << f_min << '\n';
    }

    float d = f_max - f_min + 0.001;
    float pressure = 4;
    fit_avg = 0;
    for (int i = 0; i < n; i++) {
        //fitness[i].first = pow((1 + (f_max - saveEval[i]) / d), pressure);
        //fitness[i].first = 1 / (saveEval[i] + 0.001);
        fitness[i].second = i;
        
        float score = (saveEval[i] - f_min) * (saveEval[i] - f_min) + 10;
        fitness[i].first = 1 / score;
        
        fit_avg += fitness[i].first;
    }

    fit_avg /= n;
}

void select(int gen)
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
        for (int j = 0; j < N; j++)
            vv[i][j] = v[i][j];
    
    for (int i = ELITE_SIZE; i < POP_SIZE; i++) {
        float r = rand_float(0.00001, 1);
        int j = 0;
        for (; j < n; j++)
            if (wheel[j] < r && r <= wheel[j + 1])
                break;

        for (int k = 0; k < N; k++)
            v[i][k] = vv[j][k];

        if (fitness[j].first > fit_avg)
            mutate_chromosome(v[i], mut_prob, 1);
        else
            mutate_chromosome(v[i], mut_prob, 2);
    }

    sort(fitness, fitness + n, sort_by_fitness);
    for (int i = 0; i < ELITE_SIZE; i++) {
        for (int k = 0; k < N; k++) 
            v[i][k] = vv[fitness[i].second][k];

        int r = rand_int(ELITE_SIZE, POP_SIZE - 1);
        for (int k = 0; k < N; k++) 
            swap(v[i][k], v[r][k]);
    }
    
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
                crossover_OX(parent, i);
                parent = -1;
            }
        }
    }
}

void init_population()
{
    for (int i = 0; i < POP_SIZE; i++)
        gen_greedy_instance(v[i], rand_int(0, N - 1));

    for (int i = 0; i < LOWER_SIZE; i++)
        rand_chromosome(v[rand_int(0, POP_SIZE - 1)]);
}

void evolve()
{
    //rand_population();
    rand_greedy_population();
    //init_population();

    evaluate(0);

    for (int g = 0; g < MX_GEN; g++) {
        select(g);
        //mutate(g);
        crossover_population();
        evaluate(g);
    }
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

long long current_time() 
{
    milliseconds ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    );

    return ms.count();
}

