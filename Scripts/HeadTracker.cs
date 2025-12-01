using Godot;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors; // Importante para DenseTensor
using System;
using System.Linq;
using System.Collections.Generic;

public partial class HeadTracker : Node
{
	private VideoCapture capture;
	private InferenceSession session;
	private Mat frame;
	private Mat resizedFrame;
	
	// Variável pública para o Player ler
	public float NosePositionX { get; private set; } = 0.5f;

	// Configuração do YOLO
	private const int ModelInputWidth = 640;
	private const int ModelInputHeight = 640;

	public override void _Ready()
	{
		// 1. Inicia Webcam (Index 0 é geralmente a padrão)
		// No Linux, se não abrir, tente index 1 ou verifique permissões (/dev/video0)
		capture = new VideoCapture(0);
		
		// Configura para ser rápido (baixa resolução é ok para IA)
		capture.Set(VideoCaptureProperties.FrameWidth, 640);
		capture.Set(VideoCaptureProperties.FrameHeight, 480);

		frame = new Mat();
		resizedFrame = new Mat();

		try 
		{
			// 2. Carrega modelo YOLO
			// Certifique-se que o arquivo .onnx está na raiz do projeto (res://)
			string modelPath = ProjectSettings.GlobalizePath("res://yolov8n-pose.onnx");
			session = new InferenceSession(modelPath);
			GD.Print("Modelo IA Carregado com Sucesso!");
		}
		catch (Exception e)
		{
			GD.PrintErr("Erro ao carregar modelo ONNX: " + e.Message);
		}
	}

	public override void _Process(double delta)
	{
		if (capture != null && capture.IsOpened())
		{
			capture.Read(frame);
			
			if (!frame.Empty())
			{
				// Espelhar a imagem para ficar natural (movimento de espelho)
				Cv2.Flip(frame, frame, FlipMode.Y);

				// 3. Pré-processamento: Resize e Normalização
				var inputTensor = ProcessImageToTensor(frame); 
				
				// 4. Inferência (Rodar a IA)
				var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
				
				using (var results = session.Run(inputs))
				{
					// 5. Pós-processamento: Achar o nariz no resultado
					// O resultado do YOLO Pose geralmente é um array float
					var output = results.First().AsTensor<float>();
					NosePositionX = ExtractNoseX(output);
				}
			}
		}
	}

	// Função Auxiliar 1: Converte Imagem OpenCV -> Tensor ONNX
	private DenseTensor<float> ProcessImageToTensor(Mat image)
	{
		// Redimensiona para 640x640 (o tamanho que o YOLOv8 exige)
		Cv2.Resize(image, resizedFrame, new OpenCvSharp.Size(ModelInputWidth, ModelInputHeight));

		// Cria o Tensor (formato 1 x 3 canais x 640 x 640)
		var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputHeight, ModelInputWidth });

		// Loop manual para preencher o tensor pixel a pixel (normalizando de 0-255 para 0.0-1.0)
		// O YOLO espera formato RGB Planar (RRR... GGG... BBB...)
		// O OpenCV usa BGR por padrão, então trocamos os canais
		for (int y = 0; y < ModelInputHeight; y++)
		{
			for (int x = 0; x < ModelInputWidth; x++)
			{
				var pixel = resizedFrame.At<Vec3b>(y, x);
				
				// Normaliza (divide por 255.0f)
				tensor[0, 0, y, x] = pixel.Item2 / 255.0f; // R (OpenCV Item2 é Red se convertermos, mas nativo é BGR... vamos assumir BGR->RGB troca simples)
				tensor[0, 1, y, x] = pixel.Item1 / 255.0f; // G
				tensor[0, 2, y, x] = pixel.Item0 / 255.0f; // B
			}
		}

		return tensor;
	}

	// Função Auxiliar 2: Lê o Tensor de saída e acha o Nariz
	private float ExtractNoseX(Tensor<float> output)
	{
		// Output do YOLOv8 Pose shape: [1, 56, 8400]
		// Dimensão 1 (56): 
		//   Indices 0-3: Box (cx, cy, w, h)
		//   Indice 4: Confiança do Objeto (Score)
		//   Indices 5-55: Keypoints (17 keypoints * 3 valores: x, y, conf)
		
		int channels = output.Dimensions[1]; // 56
		int anchors = output.Dimensions[2];  // 8400

		float maxScore = 0f;
		int bestAnchorIndex = -1;

		// Passo A: Achar a melhor detecção (maior score)
		// Varre todas as 8400 âncoras para ver qual tem maior certeza de ser uma pessoa
		for (int i = 0; i < anchors; i++)
		{
			float score = output[0, 4, i]; // Índice 4 é a confiança
			if (score > maxScore)
			{
				maxScore = score;
				bestAnchorIndex = i;
			}
		}

		// Se achou alguém com confiança mínima (ex: > 50%)
		if (bestAnchorIndex != -1 && maxScore > 0.5f)
		{
			// Passo B: Ler a posição do Nariz (Keypoint 0)
			// Keypoints começam no índice 5.
			// Nariz X = índice 5
			// Nariz Y = índice 6
			// Nariz Conf = índice 7
			
			float noseX = output[0, 5, bestAnchorIndex];
			
			// O valor retornado é em pixels (0 a 640). Vamos normalizar para 0.0 a 1.0
			float normalizedX = noseX / ModelInputWidth;
			
			// Garante que está entre 0 e 1
			return Math.Clamp(normalizedX, 0f, 1f);
		}

		// Se não achou ninguém, retorna a última posição conhecida ou centro
		return NosePositionX; 
	}

	public override void _ExitTree()
	{
		capture?.Dispose();
		session?.Dispose();
	}
}
