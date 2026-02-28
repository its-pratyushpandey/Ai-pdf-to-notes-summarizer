import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { BookOpen, FileText, Sparkles, ArrowRight, Upload, History } from 'lucide-react';
import gsap from 'gsap';
import { useEffect } from 'react';

const Home = () => {
  const navigate = useNavigate();
  const heroRef = useRef(null);

  useEffect(() => {
    // GSAP animations
    gsap.fromTo(
      '.hero-title',
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power3.out' }
    );
    gsap.fromTo(
      '.hero-subtitle',
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, delay: 0.2, ease: 'power3.out' }
    );
    gsap.fromTo(
      '.hero-cta',
      { opacity: 0, scale: 0.95 },
      { opacity: 1, scale: 1, duration: 0.6, delay: 0.4, ease: 'back.out(1.7)' }
    );
  }, []);

  const features = [
    {
      icon: <Upload className="w-6 h-6" />,
      title: 'Upload PDFs',
      description: 'Drag and drop or browse to upload your study materials'
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: 'Paste Text',
      description: 'Copy and paste any text content for instant conversion'
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      title: 'AI Processing',
      description: 'Powered by GPT-4o for intelligent note structuring'
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: 'Structured Notes',
      description: 'Get beautifully formatted notes with headings and bullets'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
      {/* Navigation */}
      <nav className="border-b border-slate-200/60 bg-white/70 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-indigo-800 rounded-xl flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-white" />
              </div>
              <span className="font-sans text-xl font-bold text-slate-900">NoteGenius</span>
            </div>
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 px-4 py-2 text-sm text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors duration-200"
              data-testid="nav-dashboard-btn"
            >
              <History className="w-4 h-4" />
              History
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 lg:py-32" ref={heroRef}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="hero-title font-sans text-5xl sm:text-6xl lg:text-7xl font-bold text-slate-900 tracking-tight mb-6">
              Transform PDFs into
              <span className="bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent"> Smart Notes</span>
            </h1>
            <p className="hero-subtitle text-lg sm:text-xl text-slate-600 mb-10 leading-relaxed max-w-2xl mx-auto">
              AI-powered note generation that converts your study materials into structured,
              easy-to-review notes in seconds.
            </p>
            <div className="hero-cta flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => navigate('/generator')}
                className="group px-8 py-4 bg-primary text-white rounded-xl font-semibold text-lg hover:bg-primary-hover shadow-lg shadow-indigo-500/20 transition-all duration-300 hover:-translate-y-0.5 flex items-center justify-center gap-2"
                data-testid="get-started-btn"
              >
                Get Started
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={() => navigate('/dashboard')}
                className="px-8 py-4 bg-white text-slate-900 rounded-xl font-semibold text-lg border border-slate-200 hover:bg-slate-50 hover:border-slate-300 shadow-sm transition-all duration-200"
                data-testid="view-history-btn"
              >
                View History
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 lg:py-24 bg-white border-t border-slate-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="font-sans text-3xl sm:text-4xl font-bold text-slate-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-slate-600">Simple, fast, and intelligent</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                viewport={{ once: true }}
                className="bg-white border border-slate-100 p-6 rounded-2xl hover:shadow-lg transition-shadow duration-300"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-100 to-blue-100 rounded-xl flex items-center justify-center text-indigo-600 mb-4">
                  {feature.icon}
                </div>
                <h3 className="font-sans text-lg font-semibold text-slate-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 lg:py-28 bg-gradient-to-br from-indigo-600 to-blue-700">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="font-sans text-4xl sm:text-5xl font-bold text-white mb-6">
            Ready to boost your productivity?
          </h2>
          <p className="text-xl text-indigo-100 mb-8">
            Start converting your materials into smart notes today.
          </p>
          <button
            onClick={() => navigate('/generator')}
            className="px-10 py-4 bg-white text-indigo-600 rounded-xl font-bold text-lg hover:bg-indigo-50 shadow-xl transition-all duration-300 hover:-translate-y-1"
            data-testid="cta-start-btn"
          >
            Create Your First Note
          </button>
        </div>
      </section>
    </div>
  );
};

export default Home;