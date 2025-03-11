import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import Editor from '@monaco-editor/react';
import { Chart, registerables } from 'chart.js';
import { Bar, Scatter } from 'react-chartjs-2';

Chart.register(...registerables);

// Default CUDA code template
const defaultCudaCode = `#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

// CPU matrix multiplication reference implementation
void matrixMultCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Basic CUDA matrix multiplication kernel
__global__ void matrixMultGPU(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Verify results
bool verifyResults(const float* cpuResult, const float* gpuResult, int N, float tolerance = 1e-5) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(cpuResult[i] - gpuResult[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i << ": CPU = " 
                      << cpuResult[i] << ", GPU = " << gpuResult[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations>" << std::endl;
        return 1;
    }
    
    int N = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]);
    
    // Allocate and initialize host memory
    size_t matrixSize = N * N * sizeof(float);
    float *h_A = (float*)malloc(matrixSize);
    float *h_B = (float*)malloc(matrixSize);
    float *h_C_CPU = (float*)malloc(matrixSize);
    float *h_C_GPU = (float*)malloc(matrixSize);
    
    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // CPU matrix multiplication
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMultCPU(h_A, h_B, h_C_CPU, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    
    // Setup kernel execution parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    // Warm-up GPU
    matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // GPU matrix multiplication timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float totalGpuTime = 0.0f;
    
    for (int iter = 0; iter < iterations; ++iter) {
        cudaEventRecord(start);
        matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalGpuTime += milliseconds;
    }
    
    float avgGpuTime = totalGpuTime / iterations;
    
    // Copy result back to host
    cudaMemcpy(h_C_GPU, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = verifyResults(h_C_CPU, h_C_GPU, N);
    
    // Output results as JSON
    std::cout << "{" << std::endl;
    std::cout << "  \\"matrix_size\\": " << N << "," << std::endl;
    std::cout << "  \\"iterations\\": " << iterations << "," << std::endl;
    std::cout << "  \\"cpu_time_ms\\": " << cpu_duration.count() << "," << std::endl;
    std::cout << "  \\"gpu_time_ms\\": " << avgGpuTime << "," << std::endl;
    std::cout << "  \\"speedup\\": " << (cpu_duration.count() / avgGpuTime) << "," << std::endl;
    std::cout << "  \\"verification\\": " << (correct ? "true" : "false") << std::endl;
    std::cout << "}" << std::endl;
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    
    return 0;
}`;

const Container = styled.div`
  width: 100%;
  max-width: 100%;
  padding: var(--space-3);
  margin: 0;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: var(--space-4);
  position: relative;
  padding-bottom: var(--space-3);
  
  &:after {
    content: '';
    position: absolute;
    width: 60px;
    height: 4px;
    background: var(--color-primary-gradient);
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    border-radius: var(--radius-full);
  }
`;

const GradientText = styled.span`
  background: var(--color-primary-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-fill-color: transparent;
`;

const Title = styled.h1`
  color: var(--color-secondary);
  margin-bottom: var(--space-2);
  font-weight: 700;
  font-size: 2rem;
  letter-spacing: -0.02em;
`;

const Subtitle = styled.p`
  color: var(--color-text-light);
  font-size: 1rem;
  max-width: 700px;
  margin: 0 auto;
  line-height: 1.5;
  font-weight: 400;
`;

const Main = styled.main`
  display: grid;
  grid-template-columns: 3fr 1fr;
  gap: var(--space-3);
  
  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
`;

const EditorSection = styled.section`
  display: flex;
  flex-direction: column;
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  background: var(--color-card);
  border: 1px solid var(--color-border);
  transition: var(--transition-normal);
  backdrop-filter: blur(10px);
  position: relative;
  height: calc(100vh - 180px);
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--color-primary-gradient);
    z-index: 1;
  }
`;

const EditorHeader = styled.div`
  background-color: var(--color-secondary);
  background-image: var(--color-secondary-gradient);
  color: white;
  padding: var(--space-3);
  display: flex;
  justify-content: space-between;
  align-items: center;
  
  h2 {
    margin: 0;
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
  }
`;

const ResultsSection = styled.section`
  display: flex;
  flex-direction: column;
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  background: var(--color-card);
  border: 1px solid var(--color-border);
  transition: var(--transition-normal);
  backdrop-filter: blur(10px);
  position: relative;
  height: calc(100vh - 180px);
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--color-primary-gradient);
    z-index: 1;
  }
`;

const ResultsHeader = styled.div`
  background-color: var(--color-secondary);
  background-image: var(--color-secondary-gradient);
  color: white;
  padding: var(--space-3);
  
  h2 {
    margin: 0;
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
  }
`;

const ControlPanel = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  padding: var(--space-3);
  background-color: rgba(248, 249, 251, 0.7);
  border-top: 1px solid var(--color-border);
  align-items: center;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  
  label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--color-text-light);
  }
`;

const Input = styled.input`
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  width: ${props => props.width || '100px'};
  transition: var(--transition-normal);
  
  &:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.15);
  }
`;

const Button = styled.button`
  background: ${props => props.disabled ? 'var(--color-gray-500)' : 'var(--color-primary-gradient)'};
  color: white;
  border: none;
  border-radius: var(--radius-md);
  padding: 0.6rem 1.25rem;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition-normal);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  box-shadow: ${props => props.disabled ? 'none' : 'var(--shadow-color-primary)'};
  position: relative;
  overflow: hidden;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(rgba(255,255,255,0.1), rgba(255,255,255,0));
    z-index: 1;
  }
  
  &:hover {
    transform: ${props => props.disabled ? 'none' : 'translateY(-1px)'};
    box-shadow: ${props => props.disabled ? 'none' : '0 6px 12px -5px rgba(67, 97, 238, 0.4)'};
  }
  
  &:active {
    transform: translateY(0);
  }
  
  &:disabled {
    cursor: not-allowed;
    opacity: 0.7;
  }
`;

const ResultsContent = styled.div`
  padding: 1rem;
  flex-grow: 1;
  overflow-y: auto;
  background-color: var(--color-card);
`;

const StatusMessage = styled.div`
  margin: 0.75rem 0;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  color: var(--color-text);
  background-color: ${props => (props.success ? 'rgba(39, 174, 96, 0.1)' : 'rgba(231, 76, 60, 0.1)')};
  border-left: 3px solid ${props => (props.success ? 'var(--color-success)' : 'var(--color-danger)')};
  display: flex;
  align-items: center;
  font-size: 0.95rem;
  
  strong {
    margin-right: 0.5rem;
    font-weight: 600;
  }
`;

const ResultRow = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 0.625rem 0;
  border-bottom: 1px solid var(--color-border);
  font-size: 0.95rem;
  
  &:last-child {
    border-bottom: none;
  }
  
  strong {
    font-weight: 600;
  }
`;

const ChartContainer = styled.div`
  margin-top: 1rem;
  height: 250px;
  padding: 0.75rem;
  background-color: rgba(236, 240, 241, 0.3);
  border-radius: var(--radius-md);
`;

const TabContainer = styled.div`
  display: flex;
  margin-bottom: var(--space-3);
  border-bottom: 1px solid var(--color-border);
  position: relative;
  padding-bottom: 2px;
  background-color: rgba(255, 255, 255, 0.7);
  border-radius: var(--radius-md);
  padding: var(--space-1);
`;

const Tab = styled.button`
  padding: var(--space-2) var(--space-4);
  font-size: 0.95rem;
  font-weight: 600;
  background-color: ${props => props.active ? 'white' : 'transparent'};
  color: ${props => props.active ? 'var(--color-primary)' : 'var(--color-text-light)'};
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: var(--transition-normal);
  position: relative;
  z-index: 1;
  box-shadow: ${props => props.active ? 'var(--shadow-sm)' : 'none'};
  flex: 1;
  
  &:hover {
    background-color: ${props => props.active ? 'white' : 'rgba(255, 255, 255, 0.7)'};
    color: var(--color-primary);
  }
`;

const BenchmarkCard = styled.div`
  border-radius: var(--radius-md);
  padding: var(--space-4);
  margin-bottom: var(--space-3);
  background-color: var(--color-card);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--color-border);
  transition: var(--transition-bounce);
  position: relative;
  overflow: hidden;
  min-height: 250px;
  width: 100%;
  display: flex;
  flex-direction: column;
  
  &:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
  }
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: var(--color-primary-gradient);
  }
`;

const BenchmarkTitle = styled.h3`
  margin-top: 0;
  margin-bottom: var(--space-3);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--color-border);
  color: var(--color-secondary);
  font-weight: 600;
  font-size: 1.2rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  
  span {
    font-size: 0.85rem;
    color: var(--color-text-light);
    font-weight: normal;
    background-color: var(--color-gray-100);
    padding: 0.2rem 0.5rem;
    border-radius: var(--radius-full);
    white-space: nowrap;
    margin-left: var(--space-2);
    flex-shrink: 0;
  }
`;

const BenchmarkMeta = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-2);
  margin-bottom: 60px; /* Add space at the bottom for the buttons */
  flex-grow: 1;
`;

const BenchmarkMetaItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  background-color: var(--color-gray-100);
  border: 1px solid var(--color-gray-200);
  align-items: center;
  font-size: 0.9rem;
  
  strong {
    font-weight: 600;
    color: var(--color-secondary);
    margin-right: var(--space-1);
  }
  
  span {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: ${props => {
      if (props.highlight === 'success') return 'var(--color-success)';
      if (props.highlight === 'danger') return 'var(--color-danger)';
      return 'var(--color-primary)';
    }};
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 60px;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: var(--space-2);
  justify-content: flex-end;
  position: absolute;
  bottom: var(--space-4);
  right: var(--space-4);
  width: calc(100% - 2 * var(--space-4));
`;

const HistoryButton = styled.button`
  padding: 0.5rem 1rem;
  background: ${props => 
    props.primary ? 'var(--color-primary-gradient)' : 
    props.danger ? 'var(--color-danger-gradient)' : 'transparent'};
  color: ${props => 
    (props.primary || props.danger) ? 'white' : 'var(--color-text)'};
  border: 1px solid ${props => 
    props.primary ? 'transparent' : 
    props.danger ? 'transparent' : 'var(--color-border)'};
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: var(--transition-normal);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 100px;
  box-shadow: ${props => 
    props.primary ? 'var(--shadow-color-primary)' : 
    props.danger ? 'var(--shadow-color-danger)' : 'none'};
  
  &:hover {
    background: ${props => 
      props.primary ? 'var(--color-primary-gradient)' : 
      props.danger ? 'var(--color-danger-gradient)' : 'rgba(236, 240, 241, 0.5)'};
    transform: translateY(-1px);
    box-shadow: ${props => 
      props.primary ? '0 6px 12px -5px rgba(67, 97, 238, 0.4)' : 
      props.danger ? '0 6px 12px -5px rgba(239, 35, 60, 0.4)' : 'none'};
  }
  
  &:active {
    transform: translateY(0);
  }
`;
const ScatterContainer = styled.div`
  height: 700px; /* Increased height to ensure scatter plot fits completely */
  margin: var(--space-4) 0;
  padding: var(--space-6); /* Increased padding for more space inside the container */
  background-color: var(--color-card);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--color-border);
  position: relative;
  transition: var(--transition-normal);
  overflow: hidden; /* Prevent content from extending outside */
  
  &:hover {
    box-shadow: var(--shadow-lg);
  }
  
  h3 {
    margin-top: 0;
    margin-bottom: var(--space-4); /* Increased margin for better spacing */
    text-align: center;
    color: var(--color-secondary);
    font-weight: 700;
    position: relative;
    display: inline-block;
    left: 50%;
    transform: translateX(-50%);
    padding-bottom: var(--space-2);
    
    &:after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 25%;
      width: 50%;
      height: 3px;
      background: var(--color-primary-gradient);
      border-radius: var(--radius-full);
    }
  }
  
  /* Ensure the chart container stays within bounds */
  canvas {
    max-height: calc(100% - 30px); /* Account for the header space */
  }
`;

function App() {
  // Load code from localStorage or use default
  const savedCode = localStorage.getItem('cudaCode');
  const savedMatrixSize = localStorage.getItem('matrixSize');
  const savedIterations = localStorage.getItem('iterations');
  const savedBenchmarkName = localStorage.getItem('benchmarkName');
  
  const [code, setCode] = useState(savedCode || defaultCudaCode);
  const [benchmarkName, setBenchmarkName] = useState(savedBenchmarkName || '');
  const [matrixSize, setMatrixSize] = useState(parseInt(savedMatrixSize) || 1024);
  const [iterations, setIterations] = useState(parseInt(savedIterations) || 10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [benchmarks, setBenchmarks] = useState([]);
  const [activeTab, setActiveTab] = useState('editor'); // 'editor' or 'history'
  
  // Save code to localStorage whenever it changes
  const handleCodeChange = (newCode) => {
    setCode(newCode);
    localStorage.setItem('cudaCode', newCode);
  };
  
  // Save other settings to localStorage
  const handleMatrixSizeChange = (value) => {
    const newSize = parseInt(value, 10) || 0;
    setMatrixSize(newSize);
    localStorage.setItem('matrixSize', newSize);
  };
  
  const handleIterationsChange = (value) => {
    const newIterations = parseInt(value, 10) || 0;
    setIterations(newIterations);
    localStorage.setItem('iterations', newIterations);
  };
  
  const handleBenchmarkNameChange = (value) => {
    setBenchmarkName(value);
    localStorage.setItem('benchmarkName', value);
  };
  
  // Fetch benchmarks from the server
  const fetchBenchmarks = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:3005/benchmarks');
      setBenchmarks(response.data);
    } catch (err) {
      console.error('Error fetching benchmarks:', err);
    }
  }, []);
  
  // Load benchmarks on initial render
  useEffect(() => {
    fetchBenchmarks();
  }, [fetchBenchmarks]);
  
  // Handle running code
  const handleRunCode = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:3005/compile', {
        code,
        size: matrixSize,
        iterations,
        name: benchmarkName
      });
      
      setResults(response.data.results);
      fetchBenchmarks(); // Refresh the benchmark list
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during execution');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Load a saved benchmark
  const loadBenchmark = (benchmark) => {
    // Update state
    setCode(benchmark.code);
    setMatrixSize(benchmark.matrix_size);
    setIterations(benchmark.iterations);
    setBenchmarkName(benchmark.name);
    setActiveTab('editor');
    
    // Save to localStorage
    localStorage.setItem('cudaCode', benchmark.code);
    localStorage.setItem('matrixSize', benchmark.matrix_size);
    localStorage.setItem('iterations', benchmark.iterations);
    localStorage.setItem('benchmarkName', benchmark.name);
  };
  
  // Delete a benchmark
  const deleteBenchmark = async (id) => {
    try {
      await axios.delete(`http://localhost:3005/benchmarks/${id}`);
      fetchBenchmarks();
    } catch (err) {
      console.error('Error deleting benchmark:', err);
    }
  };
  
  const chartData = results ? {
    labels: ['CPU', 'GPU'],
    datasets: [
      {
        label: 'Execution Time (ms)',
        data: [results.cpu_time_ms, results.gpu_time_ms],
        backgroundColor: ['rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)'],
        borderColor: ['rgb(255, 99, 132)', 'rgb(54, 162, 235)'],
        borderWidth: 1,
      },
    ],
  } : null;
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Time (ms)'
        }
      }
    }
  };
  
  // Prepare scatter plot data for matrix size vs speed
  const scatterData = {
    datasets: [
      {
        label: 'GPU Performance',
        data: benchmarks.map(b => ({
          x: b.matrix_size,
          y: b.gpu_time_ms
        })),
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: 'circle',
        order: 1
      },
      {
        label: 'CPU Performance',
        data: benchmarks.map(b => ({
          x: b.matrix_size,
          y: b.cpu_time_ms
        })),
        backgroundColor: 'rgba(255, 99, 132, 0.7)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: 'triangle',
        order: 2
      }
    ]
  };

  const scatterOptions = {
    scales: {
      x: {
        title: {
          display: true,
          text: 'Matrix Size',
          font: {
            weight: 'bold'
          }
        },
        type: 'linear',
        position: 'bottom'
      },
      y: {
        title: {
          display: true,
          text: 'Time (ms)',
          font: {
            weight: 'bold'
          }
        },
        type: 'logarithmic'
      }
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function(context) {
            const pointData = context.raw;
            return [
              `${context.dataset.label}: ${pointData.y.toFixed(2)} ms`,
              `Matrix: ${pointData.x}x${pointData.x}`,
              context.datasetIndex === 0 ? 
                `Speedup: ${benchmarks.find(b => b.matrix_size === pointData.x)?.speedup.toFixed(2)}x` : 
                ''
            ].filter(Boolean);
          }
        }
      },
      legend: {
        position: 'top',
        labels: {
          padding: 15,
          usePointStyle: true,
          pointStyle: true,
        }
      }
    },
    interaction: {
      mode: 'nearest',
      intersect: false
    },
    responsive: true,
    maintainAspectRatio: false
  };

  // Implement function to get a display name for benchmarks
  const getBenchmarkDisplayName = (benchmark) => {
    if (benchmark.name && benchmark.name.trim() !== '') {
      return benchmark.name;
    }
    return `Matrix ${benchmark.matrix_size}×${benchmark.matrix_size}`;
  };

  return (
    <Container>
      <Header>
        <Title>CUDA <GradientText>Matrix Multiplication</GradientText> Benchmark</Title>
        <Subtitle>Edit, compile, and benchmark your CUDA matrix multiplication kernels with real-time performance analysis</Subtitle>
      </Header>
      
      <TabContainer>
        <Tab 
          active={activeTab === 'editor'} 
          onClick={() => setActiveTab('editor')}
        >
          Code & Run
        </Tab>
        <Tab 
          active={activeTab === 'history'} 
          onClick={() => setActiveTab('history')}
        >
          Benchmark History
        </Tab>
      </TabContainer>
      
      {activeTab === 'editor' ? (
        <Main>
          <EditorSection>
            <EditorHeader>
              <h2>CUDA Code Editor</h2>
              <Button 
                onClick={() => {
                  setCode(defaultCudaCode);
                  localStorage.setItem('cudaCode', defaultCudaCode);
                }} 
                style={{ 
                  background: 'transparent', 
                  boxShadow: 'none', 
                  color: 'white',
                  padding: '0.4rem 0.75rem' 
                }}
              >
                Reset to Default
              </Button>
            </EditorHeader>
            <Editor
              height="100%"
              width="100%"
              defaultLanguage="cpp"
              value={code}
              onChange={handleCodeChange}
              theme="vs-dark"
              options={{
                fontSize: 14,
                minimap: { enabled: true },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
                wordWrap: 'on',
                lineNumbers: 'on',
                glyphMargin: true,
                folding: true,
                lineDecorationsWidth: 5,
                lineNumbersMinChars: 3
              }}
            />
            <ControlPanel>
              <InputGroup>
                <label htmlFor="benchmarkName">Name</label>
                <Input
                  id="benchmarkName"
                  type="text"
                  value={benchmarkName}
                  onChange={(e) => handleBenchmarkNameChange(e.target.value)}
                  placeholder="Benchmark name"
                  width="200px"
                />
              </InputGroup>
              <InputGroup>
                <label htmlFor="matrixSize">Matrix Size</label>
                <Input
                  id="matrixSize"
                  type="number"
                  value={matrixSize}
                  onChange={(e) => handleMatrixSizeChange(e.target.value)}
                  min="32"
                  max="8192"
                />
              </InputGroup>
              <InputGroup>
                <label htmlFor="iterations">Iterations</label>
                <Input
                  id="iterations"
                  type="number"
                  value={iterations}
                  onChange={(e) => handleIterationsChange(e.target.value)}
                  min="1"
                  max="100"
                />
              </InputGroup>
              <Button onClick={handleRunCode} disabled={loading} style={{ marginLeft: 'auto' }}>
                {loading ? 'Running...' : 'Compile & Run'}
              </Button>
            </ControlPanel>
          </EditorSection>
          
          <ResultsSection>
            <ResultsHeader>
              <h2>Benchmark Results</h2>
            </ResultsHeader>
            <ResultsContent>
              {loading && <p>Compiling and running CUDA code...</p>}
              
              {error && (
                <StatusMessage success={false}>
                  <strong>Error:</strong> {error}
                </StatusMessage>
              )}
              
              {results && (
                <>
                  <StatusMessage success={results.verification}>
                    <strong>Verification:</strong> {results.verification ? 'Passed ✓' : 'Failed ✗'}
                  </StatusMessage>
                  
                  <h3>Performance Metrics</h3>
                  <ResultRow>
                    <strong>Matrix Size:</strong> {results.matrix_size} x {results.matrix_size}
                  </ResultRow>
                  <ResultRow>
                    <strong>Iterations:</strong> {results.iterations}
                  </ResultRow>
                  <ResultRow>
                    <strong>CPU Time:</strong> {results.cpu_time_ms.toFixed(2)} ms
                  </ResultRow>
                  <ResultRow>
                    <strong>GPU Time:</strong> {results.gpu_time_ms.toFixed(2)} ms
                  </ResultRow>
                  <ResultRow>
                    <strong>Speedup:</strong> {results.speedup.toFixed(2)}x
                  </ResultRow>
                  
                  <h3>Performance Comparison</h3>
                  <ChartContainer>
                    {chartData && <Bar data={chartData} options={chartOptions} />}
                  </ChartContainer>
                </>
              )}
              
              {!loading && !error && !results && (
                <p>Edit your CUDA code and click "Compile & Run" to see benchmark results.</p>
              )}
            </ResultsContent>
          </ResultsSection>
        </Main>
      ) : (
        <div style={{ padding: '0 var(--space-3)' }}>
          <h2 style={{ 
            fontSize: '1.4rem', 
            marginBottom: 'var(--space-4)',
            padding: '0 var(--space-2)'
          }}>
            Benchmark History
          </h2>
          
          {benchmarks.length > 0 && (
            <ScatterContainer>
              <h3>Performance Across Matrix Sizes</h3>
              <Scatter data={scatterData} options={scatterOptions} />
            </ScatterContainer>
          )}
          
          {benchmarks.length === 0 ? (
            <p style={{ padding: '0 var(--space-2)' }}>No benchmarks have been saved yet. Run a benchmark to see it here.</p>
          ) : (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(440px, 1fr))', 
              gap: 'var(--space-4)',
              width: '100%'
            }}>
              {benchmarks.map(benchmark => (
                <BenchmarkCard key={benchmark.id}>
                  <BenchmarkTitle>
                    {getBenchmarkDisplayName(benchmark)}
                    <span>{new Date(benchmark.createdAt).toLocaleString()}</span>
                  </BenchmarkTitle>
                  
                  <BenchmarkMeta>
                    <BenchmarkMetaItem>
                      <strong>Matrix Size:</strong>
                      <span>{benchmark.matrix_size}×{benchmark.matrix_size}</span>
                    </BenchmarkMetaItem>
                    <BenchmarkMetaItem>
                      <strong>Iterations:</strong>
                      <span>{benchmark.iterations}</span>
                    </BenchmarkMetaItem>
                    <BenchmarkMetaItem>
                      <strong>CPU Time:</strong>
                      <span>{benchmark.cpu_time_ms.toFixed(2)} ms</span>
                    </BenchmarkMetaItem>
                    <BenchmarkMetaItem>
                      <strong>GPU Time:</strong>
                      <span>{benchmark.gpu_time_ms.toFixed(2)} ms</span>
                    </BenchmarkMetaItem>
                    <BenchmarkMetaItem>
                      <strong>Speedup:</strong>
                      <span>{benchmark.speedup.toFixed(2)}×</span>
                    </BenchmarkMetaItem>
                    <BenchmarkMetaItem highlight={benchmark.verification ? 'success' : 'danger'}>
                      <strong>Verification:</strong>
                      <span>{benchmark.verification ? 'Passed ✓' : 'Failed ✗'}</span>
                    </BenchmarkMetaItem>
                  </BenchmarkMeta>
                  
                  <ButtonGroup>
                    <HistoryButton primary onClick={() => loadBenchmark(benchmark)}>
                      Load Code
                    </HistoryButton>
                    <HistoryButton danger onClick={() => deleteBenchmark(benchmark.id)}>
                      Delete
                    </HistoryButton>
                  </ButtonGroup>
                </BenchmarkCard>
              ))}
            </div>
          )}
        </div>
      )}
    </Container>
  );
}

export default App;