using Godot;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;
using System.Collections.Generic;

public partial class HeadTracker : Node
{
	private InferenceSession session;
	private CameraTexture cameraTexture;
	private CameraFeed feed;
	
	// --- NOVO: Referência para a UI ---
	[Export]
	public TextureRect CameraPreview; 
	// ----------------------------------

	public float NosePositionX { get; private set; } = 0.5f;

	private const int ModelInputWidth = 640;
	private const int ModelInputHeight = 640;
	
	private int frameCounter = 0;
	private const int ProcessEveryNFrames = 2; 

	public override void _Ready()
	{
		// 1. Carregar IA
		try 
		{
			string modelPath = ProjectSettings.GlobalizePath("res://yolo11n-pose.onnx");
			session = new InferenceSession(modelPath);
			GD.Print("Modelo IA Carregado!");
		}
		catch (Exception e)
		{
			GD.PrintErr("Erro ao carregar modelo ONNX: " + e.Message);
		}

		// 2. Iniciar Webcam
		if (CameraServer.GetFeedCount() > 0)
		{
			feed = CameraServer.GetFeed(0);
			
			// Correção Linux: Definir formato antes de ativar
			var formats = feed.GetFormats();
			if (formats.Count > 0)
			{
				var parameters = (Godot.Collections.Dictionary)formats[0];
				feed.SetFormat(0, parameters);
			}

			feed.FeedIsActive = true; 
			GD.Print($"Webcam ativada: {feed.GetName()}");

			// Criar a textura
			cameraTexture = new CameraTexture();
			cameraTexture.CameraFeedId = feed.GetId();

			// --- NOVO: Ligar a textura à UI ---
			if (CameraPreview != null)
			{
				CameraPreview.Texture = cameraTexture;
			}
			// ----------------------------------
		}
		else
		{
			GD.PrintErr("Nenhuma webcam detectada!");
		}
	}

	public override void _Process(double delta)
	{
		if (cameraTexture == null || session == null) return;

		frameCounter++;
		if (frameCounter % ProcessEveryNFrames != 0) return;

		Image img = cameraTexture.GetImage();
		
		if (img == null || img.IsEmpty()) return;

		// Copiar e redimensionar para a IA
		Image imgForAi = (Image)img.Duplicate();
		imgForAi.Resize(ModelInputWidth, ModelInputHeight); 
		
		var inputTensor = ConvertGodotImageToTensor(imgForAi);

		try 
		{
			var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
			
			using (var results = session.Run(inputs))
			{
				var output = results.First().AsTensor<float>();
				NosePositionX = ExtractNoseX(output);
			}
		}
		catch (Exception e) { }
	}

	// (Funções auxiliares mantêm-se iguais)
	private DenseTensor<float> ConvertGodotImageToTensor(Image image)
	{
		var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputHeight, ModelInputWidth });
		byte[] data = image.GetData();
		
		int channels = (image.GetFormat() == Image.Format.Rgba8) ? 4 : 3;
		int pixelCount = ModelInputWidth * ModelInputHeight;
		
		if (data.Length < pixelCount * channels) return tensor;

		for (int i = 0; i < pixelCount; i++)
		{
			int dataIndex = i * channels;
			
			float r = data[dataIndex] / 255.0f;
			float g = data[dataIndex + 1] / 255.0f;
			float b = data[dataIndex + 2] / 255.0f;

			int x = i % ModelInputWidth;
			int y = i / ModelInputWidth;

			tensor[0, 0, y, x] = r;
			tensor[0, 1, y, x] = g;
			tensor[0, 2, y, x] = b;
		}

		return tensor;
	}

	private float ExtractNoseX(Tensor<float> output)
	{
		int anchors = output.Dimensions[2]; 
		float maxScore = 0f;
		int bestAnchorIndex = -1;

		for (int i = 0; i < anchors; i++)
		{
			float score = output[0, 4, i]; 
			if (score > maxScore)
			{
				maxScore = score;
				bestAnchorIndex = i;
			}
		}

		if (bestAnchorIndex != -1 && maxScore > 0.4f)
		{
			float noseX = output[0, 5, bestAnchorIndex];
			float normalizedX = noseX / ModelInputWidth;
			return 1.0f - Math.Clamp(normalizedX, 0f, 1f);
		}

		return NosePositionX; 
	}

	public override void _ExitTree()
	{
		session?.Dispose();
		if (feed != null) feed.FeedIsActive = false;
	}
}
