import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const S3ActivationVisualization = () => {
  const [data, setData] = useState([]);
  const [showDerivative, setShowDerivative] = useState(true);
  const [xRange, setXRange] = useState([-5, 5]);
  const [resolution, setResolution] = useState(100);

  // Sigmoid function
  const sigmoid = (x, derivative = false) => {
    if (derivative) {
      const sig = sigmoid(x, false);
      return sig * (1 - sig);
    }
    return 1 / (1 + Math.exp(-x));
  };

  // Softsign function
  const softsign = (x, derivative = false) => {
    if (derivative) {
      return 1 / Math.pow(1 + Math.abs(x), 2);
    }
    return x / (1 + Math.abs(x));
  };

  // S3 activation function
  const s3 = (x, derivative = false) => {
    if (x <= 0) {
      return sigmoid(x, derivative);
    } else {
      return softsign(x, derivative);
    }
  };

  // Generate data points
  useEffect(() => {
    const step = (xRange[1] - xRange[0]) / resolution;
    const newData = [];
    
    for (let i = 0; i <= resolution; i++) {
      const x = xRange[0] + i * step;
      const s3Value = s3(x, false);
      const s3Derivative = x === 0 ? null : s3(x, true);
      
      newData.push({
        x: Number(x.toFixed(3)),
        s3: Number(s3Value.toFixed(4)),
        s3_derivative: s3Derivative ? Number(s3Derivative.toFixed(4)) : null,
        sigmoid: Number(sigmoid(x, false).toFixed(4)),
        softsign: Number(softsign(x, false).toFixed(4))
      });
    }
    
    setData(newData);
  }, [xRange, resolution]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold">{`x = ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gray-50 rounded-lg">
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">
        Функция активации S3: Визуализация и анализ
      </h2>
      
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Диапазон X
          </label>
          <div className="flex space-x-2">
            <input
              type="number"
              value={xRange[0]}
              onChange={(e) => setXRange([Number(e.target.value), xRange[1]])}
              className="w-20 px-2 py-1 border border-gray-300 rounded"
              step="0.5"
            />
            <span className="py-1">до</span>
            <input
              type="number"
              value={xRange[1]}
              onChange={(e) => setXRange([xRange[0], Number(e.target.value)])}
              className="w-20 px-2 py-1 border border-gray-300 rounded"
              step="0.5"
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Разрешение
          </label>
          <input
            type="range"
            min="50"
            max="500"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
            className="w-full"
          />
          <span className="text-sm text-gray-500">{resolution} точек</span>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Отображение
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showDerivative}
              onChange={(e) => setShowDerivative(e.target.checked)}
              className="mr-2"
            />
            Показать производную
          </label>
        </div>
      </div>

      <div className="mb-8">
        <h3 className="text-lg font-semibold mb-4 text-gray-700">
          Функция S3 и её компоненты
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="x" 
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <YAxis domain={['dataMin', 'dataMax']} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="s3" 
              stroke="#2563eb" 
              strokeWidth={3}
              name="S3"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="sigmoid" 
              stroke="#dc2626" 
              strokeWidth={1}
              strokeDasharray="5 5"
              name="Sigmoid"
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="softsign" 
              stroke="#16a34a" 
              strokeWidth={1}
              strokeDasharray="5 5"
              name="Softsign"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {showDerivative && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Производная S3 (с разрывом в x=0)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis domain={[0, 0.6]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="s3_derivative" 
                stroke="#7c3aed" 
                strokeWidth={2}
                name="S3'"
                dot={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg border">
          <h4 className="font-semibold mb-3 text-gray-800">Ключевые свойства</h4>
          <ul className="space-y-2 text-sm">
            <li><strong>Область определения:</strong> ℝ</li>
            <li><strong>Область значений:</strong> (0, 1)</li>
            <li><strong>Точка перехода:</strong> x = 0, S3(0) = 0.5</li>
            <li><strong>Асимптоты:</strong> lim(x→-∞) = 0, lim(x→+∞) = 1</li>
            <li><strong>Монотонность:</strong> строго возрастающая</li>
          </ul>
        </div>
        
        <div className="bg-white p-4 rounded-lg border">
          <h4 className="font-semibold mb-3 text-gray-800">Особенности производной</h4>
          <ul className="space-y-2 text-sm">
            <li><strong>Разрыв:</strong> в точке x = 0</li>
            <li><strong>Левый предел:</strong> lim(x→0⁻) S3(x) = 0.25</li>
            <li><strong>Правый предел:</strong> lim(x→0⁺) S3(x) = 0.5</li>
            <li><strong>Затухание:</strong> экспоненциальное (x&lt;0), степенное (x&gt;0)</li>
            <li><strong>Максимум:</strong> в точке x→0⁺</li>
          </ul>
        </div>
      </div>

      <div className="mt-6 bg-white p-4 rounded-lg border">
        <h4 className="font-semibold mb-3 text-gray-800">Математическое определение</h4>
        <div className="text-sm space-y-2">
          <p><strong>S3(x) = </strong></p>
          <p className="ml-4">• sigmoid(x) = 1/(1 + e^(-x)), если x ≤ 0</p>
          <p className="ml-4">• softsign(x) = x/(1 + |x|), если x &gt; 0</p>
        </div>
      </div>
    </div>
  );
};

export default S3ActivationVisualization;