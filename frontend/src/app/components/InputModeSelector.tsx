import { Label } from "./ui/label";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";
import { ImageIcon, FolderIcon } from "lucide-react";

// Modo de entrada: "Single" imagem única ou "Batch" pasta com múltiplas imagens

// Interface
interface InputModeSelectorProps {
  value: "single" | "batch";
  onChange: (value: "single" | "batch") => void;
}

export function InputModeSelector({ value, onChange }: InputModeSelectorProps) {
  return (
    <div className="space-y-2">
      <Label>Modo de Entrada</Label>
      <RadioGroup value={value} onValueChange={(val) => onChange(val as "single" | "batch")}>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="single" id="single" />
          <Label htmlFor="single" className="flex items-center gap-2 cursor-pointer">
            <ImageIcon className="w-4 h-4" />
            Imagem Única
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="batch" id="batch" />
          <Label htmlFor="batch" className="flex items-center gap-2 cursor-pointer">
            <FolderIcon className="w-4 h-4" />
            Pasta com Múltiplas Imagens
          </Label>
        </div>
      </RadioGroup>
    </div>
  );
}
