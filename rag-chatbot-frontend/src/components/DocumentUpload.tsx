import React, { useState } from "react";
import axios from "axios";

const DocumentUpload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<string>("");

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setFile(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setStatus("Please select file to upload");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            setStatus("Uploading...");
            const response = await axios.post("http://localhost:8000/ingest", formData, {
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            });
            setStatus("Upload successful!" + JSON.stringify(response.data));
        } catch (error: any) {
            console.error("Upload failed:", error);
            setStatus("Upload failed. See console for details.");
        }
    };

    return (
        <div>
            <h2>Upload Document</h2>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload</button>
            <p>{status}</p>
        </div>
    );
};

export default DocumentUpload;