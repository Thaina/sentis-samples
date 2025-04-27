using System;
using System.Linq;

using UnityEngine;
using UnityEngine.SceneManagement;

using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.HandLandmarker;

using Unity.Mathematics;

#if UNITY_ANDROID && !UNITY_EDITOR
public enum SolvePnPFlags : int
{
    /// <summary>
    /// Iterative method is based on Levenberg-Marquardt optimization.
    /// In this case the function finds such a pose that minimizes reprojection error,
    /// that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .
    /// </summary>
    Iterative = 0,

    /// <summary>
    /// Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the paper “EPnP: Efficient Perspective-n-Point Camera Pose Estimation”.
    /// </summary>
    EPNP = 1,

    /// <summary>
    /// Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang“Complete Solution Classification for
    /// the Perspective-Three-Point Problem”. In this case the function requires exactly four object and image points.
    /// </summary>
    P3P = 2,

    /// <summary>
    /// Joel A. Hesch and Stergios I. Roumeliotis. "A Direct Least-Squares (DLS) Method for PnP"
    /// </summary>
    DLS = 3,

    /// <summary>
    /// A.Penate-Sanchez, J.Andrade-Cetto, F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation"
    /// </summary>
    UPNP = 4,
}
#else
using OpenCvSharp;
#endif

public class HandDetectionMediaPipe : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod()]
    static void Init()
    {
        Screen.sleepTimeout = SleepTimeout.NeverSleep;

        SceneManager.sceneUnloaded += (scene) => {
            if(scene.name.StartsWith("AR"))
                LoaderUtility.Deinitialize();

            LoaderUtility.Initialize();
        };
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

    static float convertPixelDataToDistanceInMeters(byte[] data, XRCpuImage.Format format)
    {
        switch (format) {
            case XRCpuImage.Format.DepthUint16:
                return BitConverter.ToUInt16(data, 0) / 1000f;
            case XRCpuImage.Format.DepthFloat32:
                return BitConverter.ToSingle(data, 0);
            default:
                throw new Exception($"Format not supported: {format}");
        }
    }

    public static float2? FindTForLength(in Ray ray,float z,in Ray ray1,float z1,float l)
    {
        // float3 p = o + d * t;
        // float3 p1 = o1 + d1 * t;

        // length(p1 - p) == l
        // length((o1 + d1 * (z1 + t)) - (o + d * (z + t))) == l
        // length((o1 + (d1 * z1) + (d1 * t)) - (o + (d * z) + (d * t))) == l
        // length((o1 + (d1 * z1) - o - (d * z)) + (d1 * t) - (d * t))) == l
        // length((o1 - o + (d1 * z1) - (d * z)) + t * (d1 - d))) == l
        float3 o1 = ray1.origin - ray.origin + (ray1.direction * z1) - (ray.direction * z);
        float3 d1 = ray1.direction - ray.direction;

        // length(o1 + t * d1) == l
        // csum((o1 + t * d1)^2) == l^2
        // csum(o1^2 + (2 * t * d1 * o1) + (t * d1)^2) == l^2
        // csum(o1^2) + csum(2 * t * d1 * o1) + csum((t * d1)^2) == l^2
        // csum(o1^2) + 2 * t * csum(d1 * o1) + t^2 * csum((d1)^2) == l^2
        // t^2 * csum(d1^2) + t * 2 * csum(d1 * o1) + csum(o1^2) - l^2 == 0
        float a = math.csum(d1 * d1);
        float b = 2 * math.csum(d1 * o1);
        float c = math.csum(o1 * o1) - l*l;

        return SolveQuadratic(a,b,c);
    }

    public static float2? SolveQuadratic(float a,float b,float c)
    {
        float discriminant = b * b - 4 * a * c;

        // No real solutions
        if (discriminant < 0)
            return null;

        float sqrtDiscriminant = math.sqrt(discriminant);
        float t1 = (-b + sqrtDiscriminant) / (2 * a);
        float t2 = (-b - sqrtDiscriminant) / (2 * a);

        return new float2(t1, t2);
    }

    public float[] extends = new float[] { 0.1f,0.1f,0.09f };
    public int2[] fingerPairIndex = new [] { new int2(0,9),new int2(0,5),new int2(5,17) };

    public async void Start()
    {
        var baseOptions = new BaseOptions(modelAssetBuffer: handLandmarkModel.bytes);
        var handLandmarker = HandLandmarker.CreateFromOptions(new HandLandmarkerOptions(baseOptions,runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.VIDEO));

        var depthTmpTexture = new Texture2D(256,256,TextureFormat.R16,false);
        var displayMatrix = Matrix4x4.identity;
        cameraManager.frameReceived += (args) => {
            if(args.projectionMatrix is {} _projectionMatrix)
                Camera.main.projectionMatrix = _projectionMatrix;

            if(args.displayMatrix is {} _displayMatrix)
                displayMatrix = _displayMatrix;
        };

        var imageTexture = new Texture2D(256,256,TextureFormat.RGBA32,false);
        while(this)
        {
            var solvePnPFlags = SolvePnPFlags.UPNP;

            try
            {
                await Awaitable.NextFrameAsync();
                if(!(cameraManager && cameraManager.TryAcquireLatestCpuImage(out var _image)))
                    continue;

                using var image = _image;

                if((imageTexture.width,imageTexture.height) != (image.width,image.height))
                    imageTexture.Reinitialize(image.width,image.height);

                var conversionParams = new XRCpuImage.ConversionParams(image, imageTexture.format);

                image.Convert(conversionParams,imageTexture.GetRawTextureData<byte>());
                imageTexture.Apply();

                var result = default(HandLandmarkerResult);
                if(!handLandmarker.TryDetectForVideo(new Mediapipe.Image(imageTexture),TimeSpan.FromSeconds(Time.realtimeSinceStartupAsDouble),null,ref result))
                    continue;

                lastResult = result;

                var camera = Camera.main;
                float screenRatio = Screen.width / (float)Screen.height;
                var rot = Quaternion.Euler(GizmoRot) * camera.transform.rotation;
                foreach(var flat in result.handLandmarks)
                {
                    // Convert projection matrix to camera matrix format
                    var projMatrix = camera.projectionMatrix;

                    var flatLandmarks = flat.landmarks.Select((landmark) => {
                        float3 point = (Vector3)landmark;

                        point.xy = (point.xy * 2) - 1;

                        // Apply display matrix transformation
                        point.xy = math.transform(displayMatrix,new float3(point.xy,1)).xy;
#if !UNITY_EDITOR
                        point.x /= screenRatio;
#endif
                        point.xy = (point.xy + 1) / 2;

                        return point;
                    }).ToArray();

                    var rays = flatLandmarks.Select((point) => {
                        return camera.ViewportPointToRay(point);
                    }).ToArray();

                    var triangleRays = new [] { rays[0],rays[5],rays[17] };

                    float e = fingerPairIndex.Select((fingerIndex,i) => {
                        var (m,n) = (fingerIndex[0],fingerIndex[1]);
                        var f2t = FindTForLength(rays[m],flatLandmarks[m].z,rays[n],flatLandmarks[n].z,extends[i]);
                        return f2t;
                    }).OfType<float2>().SelectMany((f2t) => new [] { f2t[0],f2t[1] }).OrderBy((e) => {
                        return fingerPairIndex.Select((fingerIndex,i) => {
                            var (m,n) = (fingerIndex[0],fingerIndex[1]);
                            return math.abs(math.distance(rays[m].GetPoint(flatLandmarks[m].z + e),rays[n].GetPoint(flatLandmarks[n].z + e)) - extends[i]);
                        }).Sum();
                    }).DefaultIfEmpty(1).FirstOrDefault();

                    var worldLandmarks = rays.Select((ray,i) => {
                        return ray.GetPoint(flatLandmarks[i].z + e);
                    }).ToArray();

                    // Create transformation matrix
                    var transform = Matrix4x4.identity;

#if UNITY_ANDROID && !UNITY_EDITOR
                    projMatrix = projMatrix.transpose;

                    using var proxyClass = new AndroidJavaClass("com.unity.opencvutils.Proxy");

                    // Convert 3D points to flattened array
                    var points3D = worldLandmarks.SelectMany(p => new double[] { p.x, p.y, p.z }).ToArray();
                    // Convert 2D points to flattened array
                    var points2D = flatLandmarks.SelectMany(p => new double[] { p.x * imageTexture.width, p.y * imageTexture.height }).ToArray();
                    // Convert camera matrix to double array
                    var cameraProjectionMatrix = Enumerable.Range(0,16).Select((i) => (double)projMatrix[i]).ToArray();

                    var transformMatrix = proxyClass.CallStatic<double[]>("solvePnP", points3D, points2D, cameraProjectionMatrix,
                        (double)imageTexture.width, (double)imageTexture.height, false, (int)solvePnPFlags);

                    // Copy result to transform matrix
                    for(int i = 0; i < 16; i++)
                        transform[i / 4, i % 4] = (float)transformMatrix[i];
#else
                    var cameraMatrix = new double[3,3];
                    cameraMatrix[0,0] = projMatrix[0, 0] * imageTexture.width * 0.5f;
                    cameraMatrix[1,1] = projMatrix[1, 1] * imageTexture.height * 0.5f;
                    cameraMatrix[0,2] = imageTexture.width * 0.5f;
                    cameraMatrix[1,2] = imageTexture.height * 0.5f;
                    cameraMatrix[2,2] = 1;

                    // Convert landmarks to OpenCV format
                    var objPoints = worldLandmarks.Select(v => new Point3f(v.x, v.y, v.z));

                    var imgPoints = flatLandmarks.Select(l => new Point2f(l.x * imageTexture.width, l.y * imageTexture.height));

                    // Create distortion coefficients (k1, k2, p1, p2, k3)
                    var distCoeffs = new double[] {
                        0.1, // k1 - radial distortion
                        0.2, // k2
                        0.01, // p1 - tangential distortion
                        0.01, // p2
                        0.3  // k3
                    };

                    var rvec = new double[3];
                    var tvec = new double[3];
                    Cv2.SolvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, ref rvec, ref tvec, false, solvePnPFlags);

                    // Convert rotation vector to matrix
                    Cv2.Rodrigues(rvec, out var rotMatrix,out var jacobian);

                    for(int i = 0; i < 3; i++)
                        for(int j = 0; j < 3; j++)
                            transform[i, j] = (float)rotMatrix[i,j];

                    transform[0, 3] = (float)tvec[0];
                    transform[1, 3] = (float)tvec[1];
                    transform[2, 3] = (float)tvec[2];
#endif
                    for(var i = 0; i < worldLandmarks.Length; i++)
                    {
                        var worldLandmark = worldLandmarks[i];
                        var position = transform.MultiplyPoint(worldLandmark);
                        position = camera.transform.TransformPoint(position);
                        handPreview.SetKeypoint(i, true, position);

                        if(i == 0 && distance)
                            distance.text = math.distance(worldLandmark,camera.transform.position).ToString("0.00");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch(Exception e)
            {
                Debug.LogException(e);
                throw;
            }
        }
    }

    public float scale = 1;

    public TMPro.TMP_Text distance;

    HandLandmarkerResult lastResult;
	void OnDrawGizmos()
	{
        if(!(lastResult.handLandmarks?.Count > 0))
            return;

        var camera = Camera.main;
        var rot = Quaternion.Euler(GizmoRot) * camera.transform.rotation;
        foreach(var (flat,world) in lastResult.handLandmarks.Zip(lastResult.handWorldLandmarks,(outer,inner) => (outer,inner)))
        {
            var worldWristPos = (Vector3)world.landmarks[0];

            var wristRay = camera.ViewportPointToRay((Vector3)flat.landmarks[0]);

            var wristAssume = wristRay.GetPoint(1);

            for(var i = 0; i < flat.landmarks.Count; i++)
            {
                var point = flat.landmarks[i];
                var landmarkRay = camera.ViewportPointToRay((Vector3)point);

                var worldLandmark = (rot * (((Vector3)world.landmarks[i] - worldWristPos) * GizmoScale)) + wristAssume;

                Gizmos.DrawRay(landmarkRay);

                Gizmos.DrawCube(worldLandmark,0.01f * Vector3.one);
            }

            foreach(var p in flat.landmarks.Select((landmark) => camera.ViewportPointToRay((Vector3)landmark).GetPoint(landmark.z + 1)))
            {
                Gizmos.DrawSphere(p,0.01f);
            }
        }
    }

    public float3 GizmoPos;
    public float3 GizmoRot;
    public float GizmoScale = 5;

	void OnDestroy()
    {
        m_DetectAwaitable?.Cancel();
    }
}
