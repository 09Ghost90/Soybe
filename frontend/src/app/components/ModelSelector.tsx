import { ZapIcon, TargetIcon, RocketIcon, LayersIcon, SmartphoneIcon } from "lucide-react";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const models = [
    { 
      id: "EfficientNetB0", 
      name: "EfficientNet-B0",
      description: "Desempenho Veloz",
      icon: ZapIcon,
      color: "text-amber-500",
      bgHover: "hover:bg-amber-50" 
    },
    { 
      id: "EfficientNetB7", 
      name: "EfficientNet-B7",
      description: "Precisão Máxima",
      icon: TargetIcon,
      color: "text-blue-500",
      bgHover: "hover:bg-blue-50" 
    },
    { 
      id: "ResNet50", 
      name: "ResNet-50",
      description: "Arquitetura Clássica",
      icon: LayersIcon,
      color: "text-purple-500",
      bgHover: "hover:bg-purple-50" 
    },
    { 
      id: "MobileNetV3", 
      name: "MobileNet-V3",
      description: "Leve e Eficiente",
      icon: SmartphoneIcon,
      color: "text-teal-500",
      bgHover: "hover:bg-teal-50" 
    },
  ];

  return (
    <div className="space-y-3">
      <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
        <RocketIcon className="w-4 h-4 text-gray-500" />
        Motor de IA Avançado
      </label>
      <div className="grid grid-cols-2 gap-4">
        {models.map((model) => {
          const Icon = model.icon;
          const isSelected = value === model.id;
          return (
            <div
              key={model.id}
              onClick={() => onChange(model.id)}
              className={`cursor-pointer rounded-xl border-2 p-4 flex flex-col gap-1 transition-all duration-300 ${
                isSelected
                  ? "border-green-500 bg-gradient-to-br from-green-50 to-emerald-50 text-green-900 shadow-md transform scale-[1.02]"
                  : `border-gray-100 bg-white text-gray-500 hover:border-gray-200 ${model.bgHover}`
              }`}
            >
              <div className="flex justify-between items-start mb-1">
                <Icon className={`w-6 h-6 ${isSelected ? "text-green-600" : model.color}`} />
                <div className={`w-3 h-3 rounded-full transition-colors ${isSelected ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" : "bg-gray-200"}`}></div>
              </div>
              <span className="font-bold text-sm leading-tight text-gray-800">{model.name}</span>
              <span className="text-xs font-medium text-gray-500">{model.description}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}