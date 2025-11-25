#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <fstream> // For file I/O
#include <string>  // For using std::string
#include <sstream> // For building strings
#include <iomanip> // For std::setw and std::setfill

__constant__ float vel_set[9][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, -1, 0}};
__constant__ float vel_set_weight[9] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

void export_to_vtk(const std::string &filename, int Nx, int Ny, const float *h_ux, const float *h_uy, const float *h_rho, const float *h_f)
{
    std::ofstream vtk_file(filename.c_str());
    if (!vtk_file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    const int num_points = Nx * Ny;

    // VTK Header
    vtk_file << "# vtk DataFile Version 3.0" << std::endl;
    vtk_file << "LBM Lid-Driven Cavity Simulation" << std::endl;
    vtk_file << "ASCII" << std::endl;
    vtk_file << "DATASET STRUCTURED_POINTS" << std::endl;

    // Grid Geometry
    vtk_file << "DIMENSIONS " << Nx << " " << Ny << " 1" << std::endl;
    vtk_file << "ORIGIN 0 0 0" << std::endl;
    vtk_file << "SPACING 1 1 1" << std::endl;
    vtk_file << "POINT_DATA " << num_points << std::endl;

    // Density Data
    vtk_file << "SCALARS rho float 1" << std::endl;
    vtk_file << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < num_points; ++i)
    {
        vtk_file << h_rho[i] << "\n";
    }

    // Velocity Data
    vtk_file << "VECTORS velocity float" << std::endl;
    for (int i = 0; i < num_points; ++i)
    {
        vtk_file << h_ux[i] << " " << h_uy[i] << " 0.0\n";
    }

    // Distribution Functions Data
    for (int q = 0; q < 9; ++q)
    {
        vtk_file << "SCALARS f_" << q << " float 1" << std::endl;
        vtk_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < num_points; ++i)
        {
            vtk_file << h_f[q * num_points + i] << "\n";
        }
    }

    vtk_file.close();
}

__global__ void stream_populations(int Nx, int Ny, float *d_f, float *d_faux)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int source_idx_x = idx % Nx;
    // int source_idx_y = idx / Nx;
    if (idx <= (Nx * Ny) - 1)
    {
        for (int i = 1; i < 9; i++)
        {
            // int target_idx_x = (source_idx_x + int(vel_set[i][0]) + Nx) % Nx;
            // int target_idx_y = (source_idx_y + int(vel_set[i][1]) + Ny) % Ny;
            // // // Streaming em x
            // d_f[i * Nx * Ny + idx] = d_faux[source_idx_y * Nx + target_idx_x];
            // // // Streaming em y
            // d_f[i * Nx * Ny + idx] = d_faux[target_idx_y * Nx + source_idx_x];

            int source_x = (idx % Nx - (int)vel_set[i][0] + Nx) % Nx;
            int source_y = (idx / Nx - (int)vel_set[i][1] + Ny) % Ny;
            int source_idx = source_y * Nx + source_x;
            d_f[i * Nx * Ny + idx] = d_faux[i * Nx * Ny + source_idx];
        }
    }
}

__global__ void collide_populations(int Nx, int Ny, float tau, float *d_f, float *d_feq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= (Nx * Ny) - 1)
    {
        for (int i = 1; i < 9; i++)
        {
            d_f[i * Nx * Ny + idx] += (d_feq[i * Nx * Ny + idx] - d_f[i * Nx * Ny + idx]) / tau;
        }
    }
}

__global__ void initialize_equilibrium(int Nx, int Ny, float *d_feq, float *ux, float *uy, float *rho)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= (Nx * Ny) - 1)
    {
        for (int i = 0; i < 9; i++)
        {
            float H_0 = 1.0;
            float H_1 = 3.0 * (vel_set[i][0] * ux[idx] + vel_set[i][1] * uy[idx]);
            float H_2 = 4.5 * (pow(ux[idx], 2) * (pow(vel_set[i][0], 2) - 1 / 3) + 2 * ux[idx] * uy[idx] * (vel_set[i][0] * vel_set[i][1]) + pow(uy[idx], 2) * (pow(vel_set[i][1], 2) - 1 / 3));
            d_feq[i * Nx * Ny + idx] = vel_set_weight[i] * rho[idx] * (H_0 + H_1 + H_2);
        }
    }
}

__device__ void apply_bc(int Nx, int Ny, int idx, int idx_x, int idx_y, float *d_f, float u_lid)
{
    if (idx_x == 0)
    {
        d_f[1 * Nx * Ny + idx] = d_f[3 * Nx * Ny + idx];
        d_f[5 * Nx * Ny + idx] = d_f[7 * Nx * Ny + idx];
        d_f[8 * Nx * Ny + idx] = d_f[6 * Nx * Ny + idx];
    }
    if (idx_x == Nx - 1)
    {
        d_f[3 * Nx * Ny + idx] = d_f[1 * Nx * Ny + idx];
        d_f[7 * Nx * Ny + idx] = d_f[5 * Nx * Ny + idx];
        d_f[6 * Nx * Ny + idx] = d_f[8 * Nx * Ny + idx];
    }
    if (idx_y == 0)
    {
        d_f[2 * Nx * Ny + idx] = d_f[4 * Nx * Ny + idx];
        d_f[5 * Nx * Ny + idx] = d_f[7 * Nx * Ny + idx];
        d_f[6 * Nx * Ny + idx] = d_f[8 * Nx * Ny + idx];
    }
    if (idx_y == Ny - 1)
    {
        const float rho_w = d_f[0 * Nx * Ny + idx] + d_f[1 * Nx * Ny + idx] + d_f[3 * Nx * Ny + idx] + 2 * (d_f[2 * Nx * Ny + idx] + d_f[5 * Nx * Ny + idx] + d_f[6 * Nx * Ny + idx]);
        d_f[4 * Nx * Ny + idx] = d_f[2 * Nx * Ny + idx];
        d_f[7 * Nx * Ny + idx] = d_f[5 * Nx * Ny + idx] + 0.5 * (d_f[1 * Nx * Ny + idx] - d_f[3 * Nx * Ny + idx] - rho_w * u_lid);
        d_f[8 * Nx * Ny + idx] = d_f[6 * Nx * Ny + idx] - 0.5 * (d_f[1 * Nx * Ny + idx] - d_f[3 * Nx * Ny + idx] - rho_w * u_lid);
    }
}

__global__ void apply_boundary_conditions(int Nx, int Ny, float *d_f, float u_lid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_x = idx % Nx;
    int idx_y = idx / Nx;
    if ((idx_x > 0 && idx_x < Nx - 1 && idx_y > 0 && idx_y < Ny - 1))
    {
        return;
    }
    else
    {
        apply_bc(Nx, Ny, idx, idx_x, idx_y, d_f, u_lid);
    }
}

__global__ void reconstruct_macros(int Nx, int Ny, float *ux, float *uy, float *rho, float *d_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double rho_val = 0, ux_val = 0, uy_val = 0;
    for (int q = 0; q < 9; q++)
    {
        rho_val += d_f[q * Nx * Ny + idx];
        ux_val += d_f[q * Nx * Ny + idx] * vel_set[q][0];
        uy_val += d_f[q * Nx * Ny + idx] * vel_set[q][1];
    }
    rho[idx] = rho_val;
    ux[idx] = ux_val / rho_val;
    uy[idx] = uy_val / rho_val;
}

__host__ int main(void)
{
    // Define inputs/parameters
    const int Nx = 100;
    const int Ny = 100;
    const int N_vel_set_components = 9;

    const int n_timesteps = 10000;
    const int export_frequency = 100;
    const int Re = 10;

    const float u_lid = 0.005;
    const float viscosity = u_lid * Nx / Re;
    const float tau = 0.5 + 3 * viscosity;

    const int N_nodes = Nx * Ny;
    const int f_elements = Nx * Ny * N_vel_set_components;

    const size_t u_rho_bytes = N_nodes * sizeof(float);
    const size_t f_bytes = f_elements * sizeof(float);

    // Alocar as populações na CPU
    // float *h_f, *h_rho, *h_ux, *h_uy; // Host vectors
    float *h_f = new float[f_elements];
    float *h_ux = new float[N_nodes];
    float *h_uy = new float[N_nodes];
    float *h_rho = new float[N_nodes];
    float *d_f, *d_faux, *d_feq, *d_rho, *d_ux, *d_uy; // Device vectors

    // Initialize vectors
    for (int i = 0; i < N_nodes; ++i)
    {
        h_ux[i] = 0.0f;
        h_uy[i] = 0.0f;
        h_rho[i] = 1.0f;
    }

    // h_f = (float *)malloc(f_bytes);
    // h_ux = (float *)malloc(u_rho_bytes);
    // h_uy = (float *)malloc(u_rho_bytes);
    // h_rho = (float *)malloc(u_rho_bytes);

    cudaMalloc((void **)&d_f, f_bytes);
    cudaMalloc((void **)&d_faux, f_bytes);
    cudaMalloc((void **)&d_feq, f_bytes);
    cudaMalloc((void **)&d_rho, u_rho_bytes);
    cudaMalloc((void **)&d_ux, u_rho_bytes);
    cudaMalloc((void **)&d_uy, u_rho_bytes);

    // cudaMemcpy(d_f, h_f, f_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ux, h_ux, u_rho_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, h_uy, u_rho_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, h_rho, u_rho_bytes, cudaMemcpyHostToDevice);

    const int number_of_threads = 64;
    const int number_of_blocks = (Nx * Ny + number_of_threads - 1) / number_of_threads;

    initialize_equilibrium<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_feq, d_ux, d_uy, d_rho);

    // MAIN LOOP
    for (int iteration = 0; iteration < n_timesteps; iteration++)
    {
        initialize_equilibrium<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_f, d_ux, d_uy, d_rho);
        reconstruct_macros<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_ux, d_uy, d_rho, d_f);
        if (iteration % export_frequency == 0)
        {
            std::cout << "Iteration: " << iteration << "/" << n_timesteps << " - Exporting data..." << std::endl;
            cudaMemcpy(h_ux, d_ux, u_rho_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy, d_uy, u_rho_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_rho, d_rho, u_rho_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_f, d_f, f_bytes, cudaMemcpyDeviceToHost);

            // Generate filename with zero-padding
            std::stringstream ss;
            ss << "lbm_output_" << std::setw(6) << std::setfill('0') << iteration << ".vtk";
            // Call the export function
            export_to_vtk(ss.str(), Nx, Ny, h_ux, h_uy, h_rho, h_f);
        }
        stream_populations<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_f, d_feq);
        apply_boundary_conditions<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_f, u_lid);
        reconstruct_macros<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_ux, d_uy, d_rho, d_f);
        initialize_equilibrium<<<number_of_blocks, number_of_threads>>>(Nx, Ny, d_feq, d_ux, d_uy, d_rho);
        collide_populations<<<number_of_blocks, number_of_threads>>>(Nx, Ny, tau, d_f, d_feq);
    }
    // Sincronizar as threads
    cudaDeviceSynchronize();

    // Liberar memórias
    delete[] h_ux;
    delete[] h_uy;
    delete[] h_rho;
    delete[] h_f;

    cudaFree(d_f);
    cudaFree(d_faux);
    cudaFree(d_feq);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);
}
