import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const models = [
    // Exemplo de modelos treinados disponíveis
    // Retornar o modelo
    { id: "EfficientNetB0", name: "EfficientNet-B0 (Precisão: 93.1%)" },
  ];


  // Retorna a string do modelo selecionado
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