import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import Home from './pages/Home';
import Generator from './pages/Generator';
import Dashboard from './pages/Dashboard';
import '@/index.css';

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/generator" element={<Generator />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" richColors />
    </>
  );
}

export default App;