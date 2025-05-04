import React, { useState } from "react";
import axios from "axios";

const Chat: React.FC = () => {
    const [question, setQuestion] = useState<string>("");
    const [answer, setAnswer] = useState<string>("");

    const handleAsk = async () => {
        if (!question) {
            setAnswer("Please enter a question.");
            return;
        }

        try {
            const response = await axios.post("http://localhost:8000/chat", {
                question: question
            });
            setAnswer(response.data.answer);
        } catch (error: any) {
            console.error("Error: ", error);
            setAnswer("Failed to get answer. See console for details");
        }
    };

    return (
        <div>
            <h2>Ask a Question</h2>
            <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Type your question here..."
                style={{ width: "300px", marginRight: "10px" }}
            />
            <button onClick={handleAsk}>Ask</button>
            <div style={{ marginTop: "20px" }}>
                <strong>Answer:</strong> {answer}
            </div>
        </div>
    );
};
export default Chat;