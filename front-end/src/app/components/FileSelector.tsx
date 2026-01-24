// Responsavel por gerenciar arquivos selecionados pelo usuario

import React from "react";

export type FileMeta = {
  file: File;
  name: string;
  size: number;
  type: string;
};

interface FileSelectorProps {
  onFilesSelected?: (files: FileMeta[]) => void;
}

export function FileSelector({ onFilesSelected }: FileSelectorProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const json = files.map(f => ({
      name: f.name,
      size: f.size,
      type: f.type
    }));

    onFilesSelected?.(json);
  };

  return (
    <input
      type="file"
      multiple
      onChange={handleChange}
    />
  );
}
