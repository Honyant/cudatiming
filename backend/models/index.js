const { Sequelize, DataTypes } = require('sequelize');
const path = require('path');

// Initialize SQLite database
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: path.join(__dirname, '../database.sqlite'),
  logging: false
});

// Define Benchmark model
const Benchmark = sequelize.define('Benchmark', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  name: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'Untitled Benchmark'
  },
  matrix_size: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  iterations: {
    type: DataTypes.INTEGER,
    allowNull: false
  },
  cpu_time_ms: {
    type: DataTypes.FLOAT,
    allowNull: false
  },
  gpu_time_ms: {
    type: DataTypes.FLOAT,
    allowNull: false
  },
  speedup: {
    type: DataTypes.FLOAT,
    allowNull: false
  },
  verification: {
    type: DataTypes.BOOLEAN,
    allowNull: false
  },
  code: {
    type: DataTypes.TEXT,
    allowNull: false
  },
  timestamp: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  }
});

// Sync models with database
sequelize.sync();

module.exports = {
  sequelize,
  Benchmark
};