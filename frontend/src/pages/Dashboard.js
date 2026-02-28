import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'sonner';
import { BookOpen, ArrowLeft, FileText, Calendar, Tag, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const navigate = useNavigate();
  const [notes, setNotes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedNote, setSelectedNote] = useState(null);

  useEffect(() => {
    fetchNotesHistory();
  }, []);

  const fetchNotesHistory = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API}/notes-history`);
      setNotes(response.data);
    } catch (error) {
      console.error('Error fetching notes:', error);
      toast.error('Failed to load notes history');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  };

  const getLengthColor = (length) => {
    const colors = {
      short: 'bg-green-100 text-green-700 border-green-200',
      medium: 'bg-blue-100 text-blue-700 border-blue-200',
      detailed: 'bg-purple-100 text-purple-700 border-purple-200'
    };
    return colors[length] || colors.medium;
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
            <button
              onClick={() => navigate('/generator')}
              className="px-4 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary-hover transition-colors"
              data-testid="new-note-btn"
            >
              New Note
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="font-sans text-3xl font-bold text-slate-900 mb-2">Notes History</h1>
          <p className="text-slate-600">View and manage all your generated notes</p>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
          </div>
        ) : notes.length === 0 ? (
          <div className="text-center py-20">
            <BookOpen className="w-16 h-16 text-slate-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 mb-2">No notes yet</h3>
            <p className="text-slate-600 mb-6">Create your first note to get started</p>
            <button
              onClick={() => navigate('/generator')}
              className="px-6 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-hover transition-colors"
              data-testid="create-first-note-btn"
            >
              Create Note
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" data-testid="notes-grid">
            {notes.map((note, index) => (
              <motion.div
                key={note.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="bg-white rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden cursor-pointer"
                onClick={() => setSelectedNote(note)}
                data-testid={`note-card-${note.id}`}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2 text-slate-600">
                      <FileText className="w-4 h-4" />
                      <span className="text-sm font-medium">
                        {note.source_type === 'pdf' ? note.original_filename : 'Text Input'}
                      </span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full border font-medium capitalize ${getLengthColor(note.notes_length)}`}>
                      {note.notes_length}
                    </span>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-slate-600 line-clamp-3">
                      {note.extracted_text.substring(0, 150)}...
                    </p>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Calendar className="w-3 h-3" />
                    {formatDate(note.created_at)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Note Preview Modal */}
      {selectedNote && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedNote(null)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
            data-testid="note-preview-modal"
          >
            <div className="border-b border-slate-200 p-6 flex justify-between items-center bg-slate-50">
              <div>
                <h3 className="font-sans text-xl font-bold text-slate-900 mb-1">
                  {selectedNote.source_type === 'pdf' ? selectedNote.original_filename : 'Text Note'}
                </h3>
                <div className="flex items-center gap-3 text-sm text-slate-600">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {formatDate(selectedNote.created_at)}
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium border capitalize ${getLengthColor(selectedNote.notes_length)}`}>
                    {selectedNote.notes_length}
                  </span>
                </div>
              </div>
              <button
                onClick={() => setSelectedNote(null)}
                className="text-slate-400 hover:text-slate-600 transition-colors"
                data-testid="close-modal-btn"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-8 overflow-y-auto max-h-[calc(85vh-120px)]" data-testid="note-content">
              <div className="prose prose-slate max-w-none font-serif">
                <ReactMarkdown>{selectedNote.notes_content}</ReactMarkdown>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
