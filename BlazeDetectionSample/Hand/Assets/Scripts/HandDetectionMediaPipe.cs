using System;
using System.Diagnostics;

using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using Unity.Mathematics;

using Mediapipe;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.HandLandmarker;

public class HandDetectionMediaPipe : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod]
    static void Init()
    {
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
    }

    public HandPreview handPreview;
    public TextAsset anchorsCSV;

    public float scoreThreshold = 0.3f;

    Awaitable m_DetectAwaitable;

    [SerializeField]
    ARCameraManager cameraManager = null;

    [SerializeField]
    TextAsset handLandmarkModel;

    public Unity.Sentis.DeviceType deviceType = Unity.Sentis.DeviceType.CPU;
    public async void Start()
    {
        var baseOptions = new BaseOptions(modelAssetBuffer: handLandmarkModel.bytes);
        var handLandmarker = HandLandmarker.CreateFromOptions(new HandLandmarkerOptions(baseOptions,runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.VIDEO));

        cameraManager.frameReceived += (args) => {
            if(args.projectionMatrix is {} matrix)
                Camera.main.projectionMatrix = matrix;
        };

        var imageTexture = new Texture2D(256,256,TextureFormat.RGBA32,false);
        while(this)
        {
            await Awaitable.NextFrameAsync();
            if(!(cameraManager && cameraManager.TryAcquireLatestCpuImage(out var image)))
                continue;

            try
            {
                var ratio = image.width / (float)image.height;
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

                var result = default(HandLandmarkerResult);
                if(!handLandmarker.TryDetectForVideo(new Image(imageTexture),(long)(image.timestamp * 1_000_000),null,ref result))
                    continue;

                foreach(var hand in result.handLandmarks)
                {
                    for (var i = 0; i < hand.landmarks.Count; i++)
                    {
                        var landmark = hand.landmarks[i];

                        var position_WorldSpace = new float3(landmark.x,landmark.y,landmark.z) - new float3(0.5f,0.5f,0);
                        position_WorldSpace.x *= ratio;
#if UNITY_EDITOR
                        position_WorldSpace = Camera.main.transform.TransformPoint(position_WorldSpace + offset);
#else
                        position_WorldSpace = Camera.main.transform.TransformPoint(math.mul(quaternion.Euler(0,0,-math.PI * 0.5f),position_WorldSpace) + offset);
#endif
                        handPreview.SetKeypoint(i, true, position_WorldSpace);
                    }
                }
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

    public float3 offset = new float3(0,0,1);

    void OnDestroy()
    {
        m_DetectAwaitable?.Cancel();
    }
}
