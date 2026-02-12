import { useRef } from "react";
import { Button } from "./ui/button";
import { UploadIcon } from "lucide-react";

// Recebe o modo de entrada e as imagens selecionadas

interface FileUploaderProps {
  mode: "single" | "batch";
  onFilesSelected: (files: FileList) => void;
}

// Componente para upload de arquivos, pode ser usado para selecionar uma única imagem ou uma pasta inteira, dependendo do modo
export function FileUploader({ mode, onFilesSelected }: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  // Frontend p/ selecionar arquivos ou pastas, dependendo do modo
  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={mode === "batch"}
        onChange={handleChange}
        className="hidden"
        {...(mode === "batch" ? { webkitdirectory: "", directory: "" } as any : {})}
      />
      <Button onClick={handleClick} className="w-full" size="lg">
        <UploadIcon className="w-5 h-5 mr-2" />
        {mode === "single" ? "Selecionar Imagem" : "Selecionar Pasta"}
      </Button>
    </div>
  );
}
