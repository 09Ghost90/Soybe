import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

// Envia para o Backend o modelo selecionado

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const models = [
    { id: "EfficientNetB0", name: "EfficientNet-B0 (Mais veloz)" },
    { id: "EfficientNetB7", name: "EfficientNet-B7 (Maior Precisão)" },
  ];

  // Implementação de envio é feita no ModelPipeline.tsx [HandleClassify] -> chamada pela App.tsx

  return (
    <div className="space-y-2">
      <Label htmlFor="model-select">Modelo Treinado</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id="model-select">
          <SelectValue placeholder="Selecione um modelo" />
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem key={model.id} value={model.id}>
              {model.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}