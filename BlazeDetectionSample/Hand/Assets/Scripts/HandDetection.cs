using System;

using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using Unity.Mathematics;
using Unity.Sentis;

public class HandDetection : MonoBehaviour
{
    public HandPreview handPreview;
    public ModelAsset handDetector;
    public ModelAsset handLandmarker;
    public TextAsset anchorsCSV;

    public float scoreThreshold = 0.3f;

    const int k_NumAnchors = 2016;
    float[,] m_Anchors;

    const int k_NumKeypoints = 21;

    Awaitable m_DetectAwaitable;

    [SerializeField]
    ARCameraManager cameraManager = null;

    public Unity.Sentis.DeviceType deviceType = Unity.Sentis.DeviceType.CPU;
    public async void Start()
    {
        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        var handDetectorModel = ModelLoader.Load(handDetector);

        // post process the model to filter scores + argmax select the best hand
        var graph = new FunctionalGraph();
        var input = graph.AddInput(handDetectorModel, 0);
        var outputs = Functional.Forward(handDetectorModel, input);
        var boxes = outputs[0]; // (1, 2016, 18)
        var scores = outputs[1]; // (1, 2016, 1)
        var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
        handDetectorModel = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        using var m_HandDetectorWorker = new Worker(handDetectorModel, deviceType);

        var handLandmarkerModel = ModelLoader.Load(handLandmarker);
        using var m_HandLandmarkerWorker = new Worker(handLandmarkerModel, deviceType);

        var handDetectorModelShape = handDetectorModel.inputs[0].shape.ToTensorShape();
        using var m_DetectorInput = new Tensor<float>(handDetectorModelShape);
        using var m_LandmarkerInput = new Tensor<float>(handLandmarkerModel.inputs[0].shape.ToTensorShape());

        cameraManager.frameReceived += (args) => {
            if(args.projectionMatrix is {} matrix)
                Camera.main.projectionMatrix = matrix;
        };

        var imageTexture = new Texture2D(256,256,TextureFormat.RGBA32,false);
        while(this)
        {
            await Awaitable.NextFrameAsync();
            if(!cameraManager.TryAcquireLatestCpuImage(out var image))
                continue;

            try
            {
                var textureSize = new Vector2Int(image.width,image.height);
                if(new Vector2Int(imageTexture.width,imageTexture.height) != textureSize)
                    imageTexture.Reinitialize(textureSize.x,textureSize.y);

                var conversionParams = new XRCpuImage.ConversionParams(image, imageTexture.format) {
#if !UNITY_EDITOR
                    transformation = XRCpuImage.Transformation.MirrorX,
                    outputDimensions = textureSize,
#endif
                };

                image.Convert(conversionParams,imageTexture.GetRawTextureData<byte>());
                imageTexture.Apply();

                m_DetectAwaitable = Detect(imageTexture,m_HandDetectorWorker,m_DetectorInput,m_HandLandmarkerWorker,m_LandmarkerInput);
                await m_DetectAwaitable;
            }
            catch (OperationCanceledException)
            {
                break;
            }
            finally
            {
                image.Dispose();
            }
        }
    }

    async Awaitable Detect(Texture texture,Worker m_HandDetectorWorker,Tensor<float> m_DetectorInput,Worker m_HandLandmarkerWorker,Tensor<float> m_LandmarkerInput)
    {
        int detectorInputSize = m_DetectorInput.shape[1];
        int landmarkerInputSize = m_LandmarkerInput.shape[1];

        var textureSize = new float2(texture.width, texture.height);

        var size = Mathf.Max(texture.width, texture.height);

        // The affine transformation matrix to go from tensor coordinates to image coordinates
        var scale = size / (float)detectorInputSize;
        var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (textureSize + new float2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

        m_HandDetectorWorker.Schedule(m_DetectorInput);

        var outputScoreAwaitable = (m_HandDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var awaiter = outputScoreAwaitable.GetAwaiter();
        while(!awaiter.IsCompleted)
            await Awaitable.NextFrameAsync();

        using var outputScore = awaiter.GetResult();
        bool scorePassesThreshold = outputScore[0] >= scoreThreshold;
        handPreview.SetActive(scorePassesThreshold);

        if (!scorePassesThreshold)
            return;

        var outputIdxAwaitable = (m_HandDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputBoxAwaitable = (m_HandDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();
        using var outputIdx = await outputIdxAwaitable;
        using var outputBox = await outputBoxAwaitable;

        var idx = outputIdx[0];

        var anchorPosition = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);

        var boxCentre_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 0], outputBox[0, 0, 1]);
        var boxSize_TensorSpace = math.max(outputBox[0, 0, 2], outputBox[0, 0, 3]);

        var kp0_TensorSpace = new float2(outputBox[0, 0, 4], outputBox[0, 0, 5]);
        var kp2_TensorSpace = new float2(outputBox[0, 0, 8], outputBox[0, 0, 9]);
        var delta_TensorSpace = kp2_TensorSpace - kp0_TensorSpace;
        var theta = math.atan2(delta_TensorSpace.y, delta_TensorSpace.x);
        var rotation = 0.5f * Mathf.PI - theta;
        boxCentre_TensorSpace += 0.5f * boxSize_TensorSpace * math.normalizesafe(delta_TensorSpace);
        boxSize_TensorSpace *= 2.6f;

        var origin2 = new float2(0.5f * landmarkerInputSize, 0.5f * landmarkerInputSize);
        var scale2 = boxSize_TensorSpace / landmarkerInputSize;
        var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(boxCentre_TensorSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(rotation)), BlazeUtils.TranslationMatrix(-origin2)));
        BlazeUtils.SampleImageAffine(texture, m_LandmarkerInput, M2);

        m_HandLandmarkerWorker.Schedule(m_LandmarkerInput);

        using var landmarks = await (m_HandLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();

        for (var i = 0; i < k_NumKeypoints; i++)
        {
            var position_ImageSpace = BlazeUtils.mul(M2, new float2(landmarks[3 * i + 0], landmarks[3 * i + 1]));

            var position_WorldSpace = new float3(position_ImageSpace - (0.5f * textureSize),landmarks[3 * i + 2]) / textureSize.y;

#if UNITY_EDITOR
            position_WorldSpace = Camera.main.transform.TransformPoint(position_WorldSpace + new float3(0,0,1));
#else
            position_WorldSpace = Camera.main.transform.TransformPoint(math.mul(quaternion.Euler(0,0,-math.PI * 0.5f),position_WorldSpace) + new float3(0,0,1));
#endif
            handPreview.SetKeypoint(i, true, position_WorldSpace);
        }
    }

    void OnDestroy()
    {
        m_DetectAwaitable?.Cancel();
    }
}
