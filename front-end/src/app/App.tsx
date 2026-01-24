import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import ModelSelector from "./components/ModelSelector";
import { InputModeSelector } from "./components/InputModeSelector";
import { FileUploader } from "./components/FileUploader";
import { ClassificationResults, ClassificationResult } from "./components/ClassificationResults";
import { Button } from "./components/ui/button";
import { Component, Loader2Icon, SproutIcon } from "lucide-react";
import { toast } from "sonner";
import { handleClassify } from "./components/ModelPipeline";

// Estado global da tela
function App() {
  const [selectedModel, setSelectedModel] = useState(""); // Modelo treinado selecionado
  const [inputMode, setInputMode] = useState<"single" | "batch">("single"); // Modo de entrada de arquivos
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]); // Arquivos selecionados pelo usuário
  const [results, setResults] = useState<ClassificationResult[]>([]); // Resultados da classificação
  const [isClassifying, setIsClassifying] = useState(false); // Classificação em execução: Sim ou Não
  
// Manipula arquivos selecionados
  const handleFilesSelected = (files: FileList) => {
    const fileArray = Array.from(files);
    setSelectedFiles(fileArray);
    toast.success(`${fileArray.length} arquivo(s) selecionado(s)`);
  };

  const runClassification = async () => {
    if (!selectedModel || selectedFiles.length === 0) return;
    setIsClassifying(true);
    try {
      const data = await handleClassify(selectedModel, selectedFiles);
      setResults(data);
      toast.success("Classificação concluída");
    } catch (err) {
      console.error(err);
      toast.error("Erro ao classificar");
    } finally {
      setIsClassifying(false);
    }
  };

// Remover quando for integrar com o backend
  const simulateClassification = (file: File): Promise<ClassificationResult> => {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simular diferentes tipos de classificação
        // Substituir por chamada API no Backend
        const classifications = [
          {
            classification: "Soja Integral de Alta Qualidade",
            confidence: 95 + Math.random() * 4,
            details: {
              category: "Grão Tipo 1",
              quality: "Excelente" as const,
            },
          },
          {
            classification: "Soja com Defeitos Leves",
            confidence: 88 + Math.random() * 7,
            details: {
              category: "Grão Tipo 2",
              quality: "Boa" as const,
              defects: ["Manchas leves"],
            },
          },
          {
            classification: "Soja com Danos Moderados",
            confidence: 82 + Math.random() * 8,
            details: {
              category: "Grão Tipo 3",
              quality: "Regular" as const,
              defects: ["Descoloração", "Fissuras"],
            },
          },
          {
            classification: "Soja Quebrada ou Danificada",
            confidence: 90 + Math.random() * 8,
            details: {
              category: "Grão Tipo 4",
              quality: "Ruim" as const,
              defects: ["Grão quebrado", "Mofo", "Insetos"],
            },
          },
        ];

        const randomClassification =
          classifications[Math.floor(Math.random() * classifications.length)];

        resolve({
          filename: file.name,
          imageUrl: URL.createObjectURL(file),
          ...randomClassification,
        });
      }, 1000 + Math.random() * 1500);
    });
  };

  // Manipula o clique no botão de classificar (handleClassify movida para ModelPipeline.tsx)

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-3">
            <SproutIcon className="w-12 h-12 text-green-700" />
            <h1 className="text-green-800">Sistema de Classificação de Grãos de Soja</h1>
          </div>
          <p className="text-gray-600">
            Utilize inteligência artificial para classificar a qualidade dos grãos de soja
          </p>
        </div>

        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Configuração da Análise</CardTitle>
            <CardDescription>
              Configure o modelo e selecione as imagens para classificação
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <ModelSelector value={selectedModel} onChange={setSelectedModel} />
              <InputModeSelector value={inputMode} onChange={setInputMode} />
            </div>

            <div className="space-y-3">
              <FileUploader mode={inputMode} onFilesSelected={handleFilesSelected} />
              {selectedFiles.length > 0 && (
                <p className="text-sm text-gray-600 text-center">
                  {selectedFiles.length} arquivo(s) selecionado(s)
                </p>
              )}
            </div>

            <Button
              onClick={runClassification}
              disabled={isClassifying || !selectedModel || selectedFiles.length === 0}
              className="w-full"
              size="lg"
            >
              {isClassifying ? (
                <>
                  <Loader2Icon className="w-5 h-5 mr-2 animate-spin" />
                  Classificando...
                </>
              ) : (
                "Iniciar Classificação"
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        {results.length > 0 && (
          <Card>
            <CardContent className="pt-6">
              <ClassificationResults results={results} />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;
