const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const { Benchmark } = require('./models');

const app = express();
const PORT = 3005;

app.use(cors());
app.use(bodyParser.json());

// Directory for CUDA files
const cudaDir = path.join(__dirname, 'cuda');
if (!fs.existsSync(cudaDir)) {
  fs.mkdirSync(cudaDir, { recursive: true });
}

// Get all benchmarks
app.get('/benchmarks', async (req, res) => {
  try {
    const benchmarks = await Benchmark.findAll({
      order: [['createdAt', 'DESC']]
    });
    res.json(benchmarks);
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// Get a specific benchmark
app.get('/benchmarks/:id', async (req, res) => {
  try {
    const benchmark = await Benchmark.findByPk(req.params.id);
    if (!benchmark) {
      return res.status(404).json({ 
        success: false, 
        error: 'Benchmark not found' 
      });
    }
    res.json(benchmark);
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// Compile, run and save benchmark
app.post('/compile', (req, res) => {
  const { code, size, iterations, name } = req.body;
  const benchmarkName = name || `MatMul ${size}x${size}`;
  
  // Write CUDA code to file
  const cudaFile = path.join(cudaDir, 'matmul.cu');
  fs.writeFileSync(cudaFile, code);
  
  // Compile CUDA code
  exec(`nvcc -arch=sm_70 -o ${path.join(cudaDir, 'matmul')} ${cudaFile}`, (compileErr, compileStdout, compileStderr) => {
    if (compileErr) {
      return res.status(400).json({ 
        success: false, 
        phase: 'compilation',
        error: compileStderr 
      });
    }
    
    // Run CUDA code
    exec(`${path.join(cudaDir, 'matmul')} ${size} ${iterations}`, (runErr, runStdout, runStderr) => {
      if (runErr) {
        return res.status(400).json({ 
          success: false, 
          phase: 'execution',
          error: runStderr 
        });
      }
      
      // Parse output and save to database
      try {
        const results = JSON.parse(runStdout);
        
        // Save to database
        Benchmark.create({
          name: benchmarkName,
          matrix_size: results.matrix_size,
          iterations: results.iterations,
          cpu_time_ms: results.cpu_time_ms,
          gpu_time_ms: results.gpu_time_ms,
          speedup: results.speedup,
          verification: results.verification,
          code: code
        }).then(savedBenchmark => {
          return res.json({
            success: true,
            results,
            benchmark: savedBenchmark
          });
        }).catch(dbError => {
          console.error('Database error:', dbError);
          return res.json({
            success: true,
            results,
            dbError: dbError.message
          });
        });
      } catch (parseErr) {
        return res.status(400).json({ 
          success: false, 
          phase: 'parsing',
          error: 'Failed to parse output',
          output: runStdout
        });
      }
    });
  });
});

// Delete a benchmark
app.delete('/benchmarks/:id', async (req, res) => {
  try {
    const benchmark = await Benchmark.findByPk(req.params.id);
    if (!benchmark) {
      return res.status(404).json({ 
        success: false, 
        error: 'Benchmark not found' 
      });
    }
    await benchmark.destroy();
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});