import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { CheckCircle2Icon, AlertCircleIcon } from "lucide-react";
import { Progress } from "./ui/progress";

/*
Estrutura

Backend resposta bruta
       ↓
App.tsx (transforma)
       ↓
ClassificationResults (exibe)
*/

export interface ClassificationResult {
  filename: string;
  imageUrl: string;        // cria isso em App com URL.createObjectURL()
  classification: string;
  confidence: number;
  details: {
    category: string;
    quality: "Excelente" | "Boa" | "Regular" | "Ruim";
    defects?: string[];
  };
}

interface ClassificationResultsProps {
  results: ClassificationResult[];
}

export function ClassificationResults({ results }: ClassificationResultsProps) {
  if (results.length === 0) {
    return null;
  }

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case "Excelente":
        return "bg-green-500";
      case "Boa":
        return "bg-blue-500";
      case "Regular":
        return "bg-yellow-500";
      case "Ruim":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getQualityIcon = (quality: string) => {
    if (quality === "Excelente" || quality === "Boa") {
      return <CheckCircle2Icon className="w-5 h-5 text-green-600" />;
    }
    return <AlertCircleIcon className="w-5 h-5 text-yellow-600" />;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3>Resultados da Classificação</h3>
        <Badge variant="secondary">{results.length} imagem(ns) analisada(s)</Badge>
      </div>

      <div className="grid gap-4">
        {results.map((result, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="truncate">{result.filename}</span>
                {getQualityIcon(result.details.quality)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <img
                    src={result.imageUrl}
                    alt={result.filename}
                    className="w-full h-48 object-cover rounded-lg border"
                  />
                </div>
                
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Classificação</p>
                    <p className="font-semibold">{result.classification}</p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-2">Confiança: {result.confidence}%</p>
                    <Progress value={result.confidence} className="h-2" />
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Categoria</p>
                    <Badge variant="outline">{result.details.category}</Badge>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Qualidade</p>
                    <Badge className={getQualityColor(result.details.quality)}>
                      {result.details.quality}
                    </Badge>
                  </div>

                  {result.details.defects && result.details.defects.length > 0 && (
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Defeitos Detectados</p>
                      <div className="flex flex-wrap gap-1">
                        {result.details.defects.map((defect, i) => (
                          <Badge key={i} variant="destructive" className="text-xs">
                            {defect}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
