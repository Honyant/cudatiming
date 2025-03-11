# CUDA Matrix Multiplication Benchmark

A web application for testing, benchmarking, and visualizing CUDA matrix multiplication kernels with CPU verification.

**Demo:**
[demo.mp4](demo.mp4)

## Features

- Edit CUDA code directly in the browser
- Compile and run CUDA kernels on your GPU
- Benchmark GPU implementation against CPU reference
- Verify correctness of computation
- Visualize performance metrics

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (nvcc must be available in PATH)
- Node.js and npm
- Modern web browser

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cudatiming.git
cd cudatiming
```

2. Install dependencies:
```bash
npm run install:all
```

## Usage

1. Start the application:
```bash
npm start
```

2. Open your browser and navigate to `http://localhost:3000`

3. Edit the CUDA kernel code in the editor, adjust parameters, and click "Compile & Run"

## Project Structure

- `backend/` - Node.js backend server
  - `index.js` - Express server for compiling and running CUDA code
  - `cuda/` - Directory for CUDA files and executables

- `frontend/` - React frontend application
  - `src/` - React components and application logic
  - `public/` - Static assets

## Development

- Backend server runs on port 3001
- Frontend development server runs on port 3000

## Examples

The application includes a default matrix multiplication implementation to get you started. You can modify this code to experiment with different optimization techniques:

1. Shared memory optimization
2. Loop unrolling
3. Thread coarsening
4. Different block/grid configurations

## License

MIT