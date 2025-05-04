import './App.css'
import DocumentUpload from './components/DocumentUpload'
import Chat from './components/Chat'

function App() {
  return (
    <div>
      <h1>RAG Chatbot</h1>
      <DocumentUpload />
      <Chat />
    </div>
  );
}

export default App;
