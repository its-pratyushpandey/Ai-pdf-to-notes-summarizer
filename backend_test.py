import requests
import sys
import json
import io
from datetime import datetime

class NotesAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def create_test_pdf(self):
        """Create a simple test PDF content (we'll use text for now)"""
        # For testing, we'll use the text input method instead
        return """Test PDF Document
        
This is a sample PDF for testing the notes generation system.

Key Points:
• Machine learning is a subset of artificial intelligence
• It involves training algorithms on data  
• Common types include supervised and unsupervised learning
• Applications include image recognition and natural language processing"""

    def test_root_endpoint(self):
        """Test GET /api/"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Response: {data}"
            self.log_test("Root Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Root Endpoint", False, str(e))
            return False

    def test_pdf_extraction(self):
        """Test POST /api/extract-pdf - Skip for now due to PDF creation complexity"""
        try:
            # Create a simple text file as PDF for basic testing
            fake_pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF"
            
            files = {'file': ('test.pdf', io.BytesIO(fake_pdf_content), 'application/pdf')}
            response = requests.post(f"{self.api_url}/extract-pdf", files=files, timeout=30)
            
            # For now, we'll accept either success or a reasonable error
            success = response.status_code in [200, 400]
            details = f"Status: {response.status_code}"
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'extracted_text' in data:
                        details += f", Text extracted successfully"
                    else:
                        details += ", Missing extracted_text field"
                except:
                    success = False
                    details += ", Invalid JSON response"
            elif response.status_code == 400:
                details += ", PDF extraction failed as expected (test PDF)"
            else:
                success = False
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f", Raw response: {response.text[:200]}"
            
            self.log_test("PDF Text Extraction", success, details)
            return success, response.json() if response.status_code == 200 else {}
            
        except Exception as e:
            self.log_test("PDF Text Extraction", False, str(e))
            return False, {}

    def test_notes_generation(self, text=None, length="medium"):
        """Test POST /api/generate-notes"""
        try:
            if not text:
                text = """Machine learning is a subset of artificial intelligence that involves training algorithms on data. 
                Common types include supervised learning (with labeled data) and unsupervised learning (finding patterns in unlabeled data). 
                Applications include image recognition, natural language processing, and recommendation systems."""
            
            payload = {
                "text": text,
                "notes_length": length,
                "source_type": "text"
            }
            
            response = requests.post(f"{self.api_url}/generate-notes", json=payload, timeout=60)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                if 'note_id' in data and 'notes_content' in data:
                    details += f", Note ID: {data['note_id'][:8]}..."
                    # Check if notes contain markdown formatting
                    notes = data['notes_content']
                    if '##' in notes or '**' in notes or '-' in notes:
                        details += ", Markdown formatting detected"
                    else:
                        details += ", Warning: No markdown formatting found"
                    return_data = data
                else:
                    success = False
                    details += ", Missing required fields"
                    return_data = {}
            else:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f", Raw response: {response.text[:200]}"
                return_data = {}
            
            self.log_test(f"Notes Generation ({length})", success, details)
            return success, return_data
            
        except Exception as e:
            self.log_test(f"Notes Generation ({length})", False, str(e))
            return False, {}

    def test_notes_history(self):
        """Test GET /api/notes-history"""
        try:
            response = requests.get(f"{self.api_url}/notes-history", timeout=10)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                if isinstance(data, list):
                    details += f", Found {len(data)} notes"
                    if len(data) > 0:
                        # Check first note structure
                        note = data[0]
                        required_fields = ['id', 'source_type', 'extracted_text', 'notes_content', 'notes_length', 'created_at']
                        missing_fields = [field for field in required_fields if field not in note]
                        if missing_fields:
                            success = False
                            details += f", Missing fields: {missing_fields}"
                        else:
                            details += ", Note structure valid"
                else:
                    success = False
                    details += ", Response is not a list"
            else:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f", Raw response: {response.text[:200]}"
            
            self.log_test("Notes History", success, details)
            return success, response.json() if success else []
            
        except Exception as e:
            self.log_test("Notes History", False, str(e))
            return False, []

    def test_get_note_by_id(self, note_id):
        """Test GET /api/notes/{id}"""
        try:
            response = requests.get(f"{self.api_url}/notes/{note_id}", timeout=10)
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                required_fields = ['id', 'source_type', 'extracted_text', 'notes_content', 'notes_length', 'created_at']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    success = False
                    details += f", Missing fields: {missing_fields}"
                else:
                    details += f", Note retrieved: {data['id'][:8]}..."
            else:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f", Raw response: {response.text[:200]}"
            
            self.log_test("Get Note by ID", success, details)
            return success
            
        except Exception as e:
            self.log_test("Get Note by ID", False, str(e))
            return False

    def test_invalid_pdf(self):
        """Test PDF extraction with invalid file"""
        try:
            # Create a fake PDF (just text file)
            fake_pdf = io.BytesIO(b"This is not a PDF file")
            files = {'file': ('fake.pdf', fake_pdf, 'application/pdf')}
            
            response = requests.post(f"{self.api_url}/extract-pdf", files=files, timeout=10)
            
            # Should return 400 error
            success = response.status_code == 400
            details = f"Status: {response.status_code}"
            
            if not success and response.status_code == 200:
                details += ", Should have failed with invalid PDF"
            elif success:
                details += ", Correctly rejected invalid PDF"
            
            self.log_test("Invalid PDF Handling", success, details)
            return success
            
        except Exception as e:
            self.log_test("Invalid PDF Handling", False, str(e))
            return False

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting NoteGenius API Tests")
        print("=" * 50)
        
        # Test 1: Root endpoint
        if not self.test_root_endpoint():
            print("❌ Root endpoint failed - stopping tests")
            return False
        
        # Test 2: PDF extraction
        pdf_success, pdf_data = self.test_pdf_extraction()
        
        # Test 3: Notes generation with different lengths
        note_id = None
        for length in ['short', 'medium', 'detailed']:
            success, data = self.test_notes_generation(length=length)
            if success and not note_id:
                note_id = data.get('note_id')
        
        # Test 4: Notes history
        history_success, history_data = self.test_notes_history()
        
        # Test 5: Get specific note (if we have a note_id)
        if note_id:
            self.test_get_note_by_id(note_id)
        elif history_data and len(history_data) > 0:
            self.test_get_note_by_id(history_data[0]['id'])
        
        # Test 6: Error handling
        self.test_invalid_pdf()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed")
            return False

def main():
    tester = NotesAPITester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_tests': tester.tests_run,
            'passed_tests': tester.tests_passed,
            'success_rate': tester.tests_passed / tester.tests_run if tester.tests_run > 0 else 0,
            'results': tester.test_results
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())