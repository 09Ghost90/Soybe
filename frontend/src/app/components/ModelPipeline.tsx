import { useState } from "react";
import ModelSelector from "./ModelSelector";
import { FileSelector, FileMeta } from "./FileSelector";

// Exported helper so App can call classification from top-level state
export async function handleClassify(model: string, files: File[]) {
    if (!model || files.length === 0) {
        throw new Error("Modelo ou arquivos ausentes");
    }

    const formData = new FormData();
    formData.append("model_name", model);
    files.forEach((f) => formData.append("files", f));

    const response = await fetch("http://localhost:8001/inferencia", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Erro na requisição: ${response.status} ${text}`);
    }

    return response.json();
}

export default function ModelPipeline() { 
    const [model, setModel] = useState("");
    const [files, setFiles] = useState<FileMeta[]>([]);
    const [isClassifying, setIsClassifying] = useState(false);
    const [results, setResults] = useState<any[]>([]);

    // Função para enviar os dados ao backend
    const sendToBackend = async () =>{
        setIsClassifying(true);

        console.log("Teste")

        const formData = new FormData();
        formData.append("model_name", model);
        
        files.forEach(fileMeta => {
            formData.append("files", fileMeta.file);
        });

        try {
            const response = await fetch("http://localhost:8001/inferencia", {
                method: "POST",
                body: formData
            });
            
        // Verifica se a resposta foi bem-sucedida
        if (!response.ok) {
            throw new Error("Erro na requisição");
        }

        const data = await response.json();
        setResults(data);
        } catch (err) {
        console.error(err);
        } finally {
        setIsClassifying(false);
        }
    };

    const handleClassify = async () => {
    if (!model || files.length === 0) return;
    await sendToBackend();
  };

    return (
        <>
        <ModelSelector value={model} onChange={setModel} />
        <FileSelector onFilesSelected={setFiles} />

        <button
            disabled={!model || files.length === 0 || isClassifying}
            onClick={handleClassify}
        >

            {isClassifying ? "Classificando..." : "Iniciar Classificação"}
        </button>

        {/* Renderiza resultados depois */}
        {results.map(r => (
            <div key={r.filename}>
            {r.filename} → {r.classification}
            </div>
        ))}
        </>
    );
}
    // Estrutura retornada para o backend
    /*
        model: "EfficientNetB0",
        files: [
            { name: "image1.jpg", size: 12345, type: "image/jpeg" },
            { name: "image2.png", size: 67890, type: "image/png" }
        ]
    }
    */