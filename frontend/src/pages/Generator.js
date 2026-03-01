import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'sonner';
import { Upload, FileText, BookOpen, Sparkles, ArrowLeft, Download, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API = `${BACKEND_URL}/api`;

const Generator = () => {
  const navigate = useNavigate();
  const [mode, setMode] = useState('pdf'); // 'pdf' or 'text'
  const [isDragging, setIsDragging] = useState(false);
  const [pdfFile, setPdfFile] = useState(null);
  const [textContent, setTextContent] = useState('');
  const [notesLength, setNotesLength] = useState('medium');
  const [isProcessing, setIsProcessing] = useState(false);
  const [generatedNotes, setGeneratedNotes] = useState('');
  const [noteId, setNoteId] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      toast.success(`Selected: ${file.name}`);
    } else {
      toast.error('Please select a PDF file');
    }
  };

  const handleGenerateNotes = async () => {
    try {
      setIsProcessing(true);
      setGeneratedNotes('');
      
      let extractedText = '';
      let filename = null;

      if (mode === 'pdf') {
        if (!pdfFile) {
          toast.error('Please upload a PDF file');
          setIsProcessing(false);
          return;
        }

        // Extract text from PDF
        const formData = new FormData();
        formData.append('file', pdfFile);
        
        const extractResponse = await axios.post(`${API}/extract-pdf`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        
        extractedText = extractResponse.data.extracted_text;
        filename = extractResponse.data.filename;
        toast.success(`Extracted ${extractResponse.data.word_count} words`);
      } else {
        if (!textContent.trim()) {
          toast.error('Please enter some text');
          setIsProcessing(false);
          return;
        }
        extractedText = textContent;
      }

      // Generate notes
      const response = await axios.post(`${API}/generate-notes`, {
        text: extractedText,
        notes_length: notesLength,
        source_type: mode,
        original_filename: filename
      });

      setGeneratedNotes(response.data.notes_content);
      setNoteId(response.data.note_id);
      toast.success('Notes generated successfully!');
    } catch (error) {
      console.error('Error generating notes:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate notes');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setPdfFile(null);
    setTextContent('');
    setGeneratedNotes('');
    setNoteId(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
      {/* Navigation */}
      <nav className="border-b border-slate-200/60 bg-white/70 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/')}
                className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
                data-testid="back-home-btn"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-indigo-800 rounded-xl flex items-center justify-center">
                  <BookOpen className="w-6 h-6 text-white" />
                </div>
                <span className="font-sans text-xl font-bold text-slate-900">NoteGenius</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 min-h-[calc(100vh-8rem)]">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 sm:p-6 flex flex-col"
            data-testid="input-panel"
          >
            <h2 className="font-sans text-2xl font-bold text-slate-900 mb-6">Create Notes</h2>
            
            {/* Mode Toggle */}
            <div className="flex gap-2 mb-6 bg-slate-100 p-1 rounded-lg">
              <button
                onClick={() => setMode('pdf')}
                className={`flex-1 py-2 px-4 rounded-md font-medium transition-all duration-200 ${
                  mode === 'pdf'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                data-testid="mode-pdf-btn"
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Upload PDF
              </button>
              <button
                onClick={() => setMode('text')}
                className={`flex-1 py-2 px-4 rounded-md font-medium transition-all duration-200 ${
                  mode === 'text'
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
                data-testid="mode-text-btn"
              >
                <FileText className="w-4 h-4 inline mr-2" />
                Paste Text
              </button>
            </div>

            {/* Input Area */}
            {mode === 'pdf' ? (
              <div
                className={`flex-1 border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all duration-200 cursor-pointer ${
                  isDragging
                    ? 'border-indigo-500 bg-indigo-50'
                    : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                data-testid="upload-dropzone"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                <Upload className="w-16 h-16 text-slate-300 mb-4" />
                {pdfFile ? (
                  <div className="text-center">
                    <p className="text-lg font-semibold text-slate-900 mb-2">{pdfFile.name}</p>
                    <p className="text-sm text-slate-500">Click to change file</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <p className="text-lg font-semibold text-slate-900 mb-2">Drop your PDF here</p>
                    <p className="text-sm text-slate-500">or click to browse</p>
                  </div>
                )}
              </div>
            ) : (
              <textarea
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                placeholder="Paste your text content here..."
                className="min-h-[240px] lg:min-h-0 flex-1 border border-slate-200 rounded-xl p-4 resize-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all duration-200"
                data-testid="text-input"
              />
            )}

            {/* Notes Length Selector */}
            <div className="mt-6">
              <label className="block text-sm font-semibold text-slate-700 mb-3">Notes Length</label>
              <div className="grid grid-cols-3 gap-2">
                {['short', 'medium', 'detailed'].map((length) => (
                  <button
                    key={length}
                    onClick={() => setNotesLength(length)}
                    className={`py-2 px-4 rounded-lg font-medium capitalize transition-all duration-200 ${
                      notesLength === length
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                    data-testid={`length-${length}-btn`}
                  >
                    {length}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerateNotes}
              disabled={isProcessing}
              className="mt-6 w-full py-4 bg-primary text-white rounded-xl font-semibold hover:bg-primary-hover shadow-lg shadow-indigo-500/20 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              data-testid="generate-btn"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating Notes...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Notes
                </>
              )}
            </button>
          </motion.div>

          {/* Output Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-background-paper rounded-2xl border border-slate-100 shadow-sm p-4 sm:p-8 flex flex-col overflow-hidden"
            data-testid="output-panel"
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="font-sans text-2xl font-bold text-slate-900">Generated Notes</h2>
              {generatedNotes && (
                <button
                  onClick={handleReset}
                  className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
                  data-testid="reset-btn"
                >
                  Start New
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto" data-testid="notes-output">
              {generatedNotes ? (
                <div className="prose max-w-none font-serif">
                  <ReactMarkdown>{generatedNotes}</ReactMarkdown>
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-center">
                  <div>
                    <BookOpen className="w-16 h-16 text-slate-300 mx-auto mb-4" />
                    <p className="text-lg text-slate-500">Your notes will appear here</p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Generator;