#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <tuple>

using namespace std;
using namespace std::chrono;

tuple<vector<double>, vector<double>, vector<double>> euler_method(double m, double k, double x0, double v0, double t_max, double dt)
{
    vector<double> t_list, x_list, v_list;
    double x = x0, v = v0, t, a;

    for (t = 0; t <= t_max; t += dt)
    {
        t_list.push_back(t);
        x_list.push_back(x);
        v_list.push_back(v);

        a = -k * x / m;
        x = x + dt * v;
        v = v + dt * a;
    }

    return make_tuple(t_list, x_list, v_list);
}

tuple<vector<double>, vector<double>, vector<double>> verlet_method(double m, double k, double x0, double v0, double t_max, double dt)
{
    vector<double> t_list, x_list, v_list;
    // initialization
    double x = x0, v = v0, t, a;
    double x_prev = x - v * dt; // Initial previous position, using first order euler method

    for (t = 0; t <= t_max; t += dt)
    {
        t_list.push_back(t);
        x_list.push_back(x);
        v_list.push_back(v);

        a = -k * x / m;
        double x_next = 2 * x - x_prev + a * dt * dt;
        v = (x_next - x_prev) / (2 * dt);

        // x_prev <- x, move one step forward
        x_prev = x;
        // x <- x_next, move one step forward
        x = x_next;
    }

    return make_tuple(t_list, x_list, v_list);
}

void write_to_file(const vector<double> &t_list, const vector<double> &x_list, const vector<double> &v_list, const string &filename)
{
    ofstream fout(filename);
    if (fout)
    {
        for (size_t i = 0; i < t_list.size(); i++)
        {
            fout << t_list[i] << ' ' << x_list[i] << ' ' << v_list[i] << endl;
        }
        cout << "Data written to " << filename << endl;
    }
    else
    {
        cout << "Could not open " << filename << " for writing" << endl;
    }
}

int main()
{
    double m = 1, k = 1, x0 = 0, v0 = 1;
    double t_max = 100, dt = 0.001;

    // Euler method
    // this method returns a time point, just use auto to infer the type for initialization
    auto start = high_resolution_clock::now();
    auto [t_euler, x_euler, v_euler] = euler_method(m, k, x0, v0, t_max, dt);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Euler method execution time: " << duration.count() / 1e6 << " seconds" << endl;

    // Verlet method
    start = high_resolution_clock::now();
    // new variable need assign auto
    auto [t_verlet, x_verlet, v_verlet] = verlet_method(m, k, x0, v0, t_max, dt);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Verlet method execution time: " << duration.count() / 1e6 << " seconds" << endl;

    // Write results to files
    write_to_file(t_euler, x_euler, v_euler, "euler_trajectories.txt");
    write_to_file(t_verlet, x_verlet, v_verlet, "verlet_trajectories.txt");

    return 0;
}