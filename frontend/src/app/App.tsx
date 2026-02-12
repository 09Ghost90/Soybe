import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import ModelSelector from "./components/ModelSelector";
import { InputModeSelector } from "./components/InputModeSelector";
import { FileUploader } from "./components/FileUploader";
import { ClassificationResults, ClassificationResult } from "./components/ClassificationResults";
import { Button } from "./components/ui/button";
import { Component, Loader2Icon, SproutIcon, StopCircleIcon, Trash } from "lucide-react";
import { toast } from "sonner";
import { handleClassify } from "./components/ModelPipeline";

// Estado global da tela
function App() {
  const [selectedModel, setSelectedModel] = useState(""); // Modelo treinado selecionado
  const [inputMode, setInputMode] = useState<"single" | "batch">("single"); // Modo de entrada de arquivos
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]); // Arquivos selecionados pelo usuário
  const [results, setResults] = useState<ClassificationResult[]>([]); // Resultados da classificação
  const [isClassifying, setIsClassifying] = useState(false); // Classificação em execução: Sim ou Não
  // Estado para armazenar o controller de cancelamento da requisição
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  
// Manipula arquivos selecionados
  const handleFilesSelected = (files: FileList) => {
    const fileArray = Array.from(files);
    setSelectedFiles(fileArray);
    toast.success(`${fileArray.length} arquivo(s) selecionado(s)`);
  };

  const runClassification = async () => {

    const controller = new AbortController();
    // Salvar oo estado do controller
    setAbortController(controller);

    // Spinner de carregamento
    setIsClassifying(true);

    if (!selectedModel || selectedFiles.length === 0) return;
    setIsClassifying(true);

    try {

      /*
      ModelPipeline (handleClassify) recebe o modelo e arquivos para classificação
      
      Data retorna a estrutura:
      {
        "filename": "image1.jpg",
        "classification": "Intact soybeans",
        "confidence": 0.98,
        "model_used": "EfficientNetB0"
      }

      data.map(item => item.filename, item.classification, item.confidence, item.model_used)

      */
      
      const data = await handleClassify(selectedModel, selectedFiles, controller.signal);

      const sortedData = data.sort((a: any, b: any) => a.index - b.index);

      console.log("Data:", sortedData);

      // Resposta do backend estruturada
      const transformedResults = sortedData.map((item: any, idx: number) => {
        // Extrai o número do filename (ex: "Broken-32/997.jpg" → 997)
        const fileNumber = parseInt(item.filename.split('/').pop()?.split('.')[0] || '0');
        
        // Encontra o arquivo correspondente em selectedFiles
        const correspondingFile = selectedFiles.find((f: File) => {
          const fNumber = parseInt(f.name.split('/').pop()?.split('.')[0] || '0');
          return fNumber === fileNumber;
        });

        if (!correspondingFile) {
          console.warn(`Arquivo ${item.filename} não encontrado em selectedFiles`);
          return null;
        }

        return {
          filename: item.filename,
          imageUrl: URL.createObjectURL(correspondingFile),
          classification: item.classification,
          confidence: Math.round((item.confidence ?? 0) * 100),
          details: {
            category: item.model_used ?? "N/A",
            quality: getQualityFromConfidence(item.confidence ?? 0),
            defects: item.classification ? [item.classification] : []
          }
        };
      }).filter(Boolean);

      console.log("Transformed Results:", transformedResults);

      setResults(transformedResults);
    } catch (err: any) {
      if (err.name === "AbortError") {
        toast.info("Classificação cancelada pelo usuário");
      } else {
        console.error("Erro:", err);
      }
    } finally {
      setIsClassifying(false);
    }
  };

// Função para chamar pelo botão "Parar"
const stopClassification = () => {
  // Cancelar a requisição HTTP se o controller existir
  abortController?.abort();
  setIsClassifying(false);
};

// Helper para mapear confidence -> quality
function getQualityFromConfidence(confidence: number): "Excelente" | "Boa" | "Regular" | "Ruim" {
  if (confidence >= 0.95) return "Excelente";
  if (confidence >= 0.85) return "Boa";
  if (confidence >= 0.70) return "Regular";
  return "Ruim";
}

// Função para limpar a seleção de arquivos e resultados
const clearSelection = () => {
  setSelectedFiles([]);
  setResults([]);
  toast.success("Seleção limpa");
}

// Função

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
            <div className="grid md:grid-cols-2 gap-6, flex items-center">
              <ModelSelector value={selectedModel} onChange={setSelectedModel} />
              <InputModeSelector value={inputMode} onChange={setInputMode} />
            </div>

              <div className="flex justify-end">
                <Button className="w-auto" onClick={clearSelection}>
                  <Trash className="w-4 h-4 mr-2" />
                  Limpar Seleção
                </Button>
              </div>

            <div className="space-y-3">
              <FileUploader mode={inputMode} onFilesSelected={handleFilesSelected} />
              {selectedFiles.length > 0 && (
                <p className="text-sm text-gray-600 text-center">
                  {/* Quantidade de arquivos selecionados */}
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

            <Button
              onClick={stopClassification}
              disabled={!isClassifying}
              className="w-full bg-red-600 hover:bg-red-800">
              <StopCircleIcon /> Parar Classificação
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
