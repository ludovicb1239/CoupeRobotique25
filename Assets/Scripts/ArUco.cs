using System.Collections.Generic;
using System.Xml.Serialization;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using OpenCVForUnity.ArucoModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine.Rendering;
using OpenCVForUnityExample;
using OpenCVForUnity.ImgprocModule;
using System;
using OpenCVForUnity.UnityUtils.Helper;
using Unity.Collections;
using System.Threading;

public enum ArucoMode
{
    Disabled,
    FakeCamera,
    RealCamera
}

class ToRunOnMainThread
{
    public ArUcoTag tag;
    public Matrix4x4 ARM;
    public int TagID;
    // maybe add time
}
public class ArUco : MonoBehaviour
{
    public RenderTexture imgTexture;

    public GameObject FakeCameraObject;
    public GameObject RealCameraObject;
    WebCamTextureToMatHelper WebCamHelper;
    public ArucoMode mode;

    [Space(10)]
    public bool debug = true;
    public bool showRejectedCorners = false;
    public bool applyEstimationPose = false;
    public Camera arCamera;

    [Space(10)]

    Texture2D texture;
    Texture2D texture2;
    public RawImage imageOut;

    private int width;
    private int height;

    Mat camMatrix;
    MatOfDouble distCoeffs;
    DetectorParameters detectorParams;
    Dictionary dictionary;

    public bool useStoredCameraParameters;

    public bool inputGray = true;

    public int smoothingPoints = 20;
    private int smoothingPointIdx;
    private Vector3[] smoothingPos;
    private Quaternion[] smoothingRot;

    public double threshold;
    public double c;

    public TextAsset xmlFile; // Reference to the XML file

    List<ToRunOnMainThread> toRunOnMainThread = new();

    public SolarPanelManagerAuto solarPanelManager;

    public Thread thread;

    // Start is called before the first frame update
    void Start()
    {
        FakeCameraObject.SetActive(mode == ArucoMode.FakeCamera);
        RealCameraObject.SetActive(mode == ArucoMode.RealCamera);
        arCamera.enabled = debug && mode != ArucoMode.Disabled;

        switch (mode)
        {
        case ArucoMode.Disabled:
            this.gameObject.SetActive(false);
            break;
        case ArucoMode.RealCamera:
            InitializeReal();
            break;
        case ArucoMode.FakeCamera:
            InitializeFake();
            break;
        }
    }

    // Update is called once per frame
    void Update()
    {
        ArUcoTag tag;
        for(int i = 0; i < toRunOnMainThread.Count; i++)
        {
            tag = toRunOnMainThread[i].tag;
            if (tag.isFixed)
            {
                toRunOnMainThread[i].ARM = tag.baseTag.transform.localToWorldMatrix * toRunOnMainThread[i].ARM.inverse;
                Transform newTransform = this.transform;
                ARUtils.SetTransformFromMatrix(newTransform, ref toRunOnMainThread[i].ARM);
                smoothingPos[smoothingPointIdx] = newTransform.position;
                smoothingRot[smoothingPointIdx] = newTransform.rotation;
                smoothingPointIdx++;
                if (smoothingPointIdx == smoothingPoints)
                    smoothingPointIdx = 0;

                arCamera.transform.position = CalculateAverage(smoothingPos);
                arCamera.transform.rotation = CalculateAverage(smoothingRot);
            }
            else
            {
                Transform tr = tag.baseTag.transform;
                toRunOnMainThread[i].ARM = arCamera.transform.localToWorldMatrix * toRunOnMainThread[i].ARM;
                ARUtils.SetTransformFromMatrix(tr, ref toRunOnMainThread[i].ARM);
                if (tag.usingRestrainPos.x || tag.usingRestrainPos.y || tag.usingRestrainPos.z)
                {
                    tr.position = new Vector3(tag.usingRestrainPos.x ? tag.restrainPosition.x : tr.position.x,
                                            tag.usingRestrainPos.y ? tag.restrainPosition.y : tr.position.y,
                                            tag.usingRestrainPos.z ? tag.restrainPosition.z : tr.position.z
                        );
                }
                if (tag.usingRestrainRot.x || tag.usingRestrainRot.y || tag.usingRestrainRot.z)
                {
                    tr.rotation = Quaternion.Euler(tag.usingRestrainRot.x ? tag.restrainRotation.x : tr.rotation.eulerAngles.x,
                                            tag.usingRestrainRot.y ? tag.restrainRotation.y : tr.rotation.eulerAngles.y,
                                            tag.usingRestrainRot.z ? tag.restrainRotation.z : tr.rotation.eulerAngles.z
                        );
                }
                if (toRunOnMainThread[i].TagID == 47) //Solar panel
                {
                    solarPanelManager.Refresh();
                }
            }
        }
        toRunOnMainThread = new();


        switch (mode)
        {
        case ArucoMode.FakeCamera:
            AsyncGPUReadback.Request(imgTexture, 0, OnReadbackComplete);
            break;
        case ArucoMode.RealCamera:

            if (WebCamHelper != null && WebCamHelper.IsPlaying() && WebCamHelper.DidUpdateThisFrame())
            {
                Mat mat = WebCamHelper.GetMat();
                thread = new Thread(() => ProcessRealImageData(mat));
                thread.Start();
            }
            break;
        }
    }

    void ProcessRealImageData(Mat mat)
    {
        if (!inputGray)
        {
            Mat grayMat = new Mat(height, width, CvType.CV_8UC3); //8UC4 8UC3
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_RGBA2GRAY);
            DetectMarkers(grayMat);
        }
        else
        {
            DetectMarkers(mat);
        }
    }

    void OnReadbackComplete(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.LogWarning("GPU readback error detected.");
            return;
        }
        if (texture2 == null)
        {
            Debug.LogWarning("Texture 2 been destroyed");
            return;
        }

        NativeArray<byte> data = request.GetData<byte>();
        // Start a new task for processing the readback data
        Thread thread = new Thread(() => ProcessReadbackData(data));
        thread.Start();
        //Task.Run(() => ProcessReadbackData(data));
    }
    void ProcessReadbackData(NativeArray<byte> data)
    {
        Mat grayMat = new Mat(height, width, CvType.CV_8UC1); //8UC4 8UC3
        Mat rgbaMat = new Mat(height, width, CvType.CV_8UC4); //8UC4 8UC3

        Utils.RawTextureDataToMat(data, rgbaMat, grayMat.width(), grayMat.height());
        Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
        rgbaMat.Dispose();
        // Dispose the readback request and release the temporary data
        data.Dispose();

        //Imgproc.adaptiveThreshold(grayMat, grayMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,(int)threshold, c);

        DetectMarkers(grayMat);
    }
    private void DetectMarkers(Mat mat)
    {
        Mat ids = new Mat();
        List<Mat> corners = new List<Mat>();
        List<Mat> rejectedCorners = new List<Mat>();

        // detect markers.
        Aruco.detectMarkers(mat, dictionary, corners, ids, detectorParams, rejectedCorners, camMatrix, distCoeffs);

        // if at least one marker detected
        Debug.Log($"Found {ids.total()} tags");
        
        if (ids.total() > 0)
        {
            if (debug)
                Aruco.drawDetectedMarkers(mat, corners, ids, new Scalar(0, 255, 200));

            // estimate pose.
            if (applyEstimationPose)
            {
                for (int i = 0; i < ids.total(); i++)
                {
                    int tagID = (int)ids.get(i, 0)[0];

                    if (!ArUcoManager.Instance.dict.ContainsKey(tagID))
                        continue;
                    ArUcoTag tag = ArUcoManager.Instance.dict[tagID];

                    EstimatePoseCanonicalMarker(mat, tag, i, tagID, corners);
                }
            }
        }


        // MainThreadDispatcher.RunOnMainThread(() =>
        // {
        //     Utils.matToTexture2D(grayMat, texture);
        // });
        if (debug)
        {
            if (showRejectedCorners && rejectedCorners.Count > 0)
                Aruco.drawDetectedMarkers(mat, rejectedCorners, new Mat(), new Scalar(255, 0, 0));

            //TODO : Run on main thread
            // Utils.matToTexture2D(mat, texture);
        }

        if (!inputGray)
            mat.Dispose();

    }
    private void EstimatePoseCanonicalMarker(Mat rgbMat, ArUcoTag tag, int i, int id, List<Mat> corners)
    {
        Mat rvecs = new Mat();
        Mat tvecs = new Mat();

        float markerLength = tag.size;
        Aruco.estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);
        using (Mat rvec = new Mat(rvecs, new OpenCVForUnity.CoreModule.Rect(0, i, 1, 1)))
        using (Mat tvec = new Mat(tvecs, new OpenCVForUnity.CoreModule.Rect(0, i, 1, 1)))
        {
            // In this example we are processing with RGB color image, so Axis-color correspondences are X: blue, Y: green, Z: red. (Usually X: red, Y: green, Z: blue)
            if (debug)
                Calib3d.drawFrameAxes(rgbMat, camMatrix, distCoeffs, rvec, tvec, markerLength); // * 0.5f

            UpdateARObjectTransform(rvec, tvec, tag, id);
        }
    }
    private void UpdateARObjectTransform(Mat rvec, Mat tvec, ArUcoTag tag, int id)
    {
        // Convert to unity pose data.
        double[] rvecArr = new double[3];
        rvec.get(0, 0, rvecArr);
        double[] tvecArr = new double[3];
        tvec.get(0, 0, tvecArr);
        PoseData poseData = ARUtils.ConvertRvecTvecToPoseData(rvecArr, tvecArr);

        // Changes in pos/rot below these thresholds are ignored.
        //if (enableLowPassFilter)
        //{
        //    ARUtils.LowpassPoseData(ref oldPoseData, ref poseData, positionLowPass, rotationLowPass);
        //}

        // Convert to transform matrix.
        Matrix4x4 ARM = ARUtils.ConvertPoseDataToMatrix(ref poseData, true);
        ToRunOnMainThread mainThread = new();
        mainThread.tag = tag;
        mainThread.ARM = ARM;
        mainThread.TagID = id;
        toRunOnMainThread.Add(mainThread);
    }

    void InitializeFake()
    {
        Application.targetFrameRate = 30;
        width = imgTexture.width;
        height = imgTexture­.height;
        FinishInitialize();
    }
    void InitializeReal()
    {
        WebCamHelper = RealCameraObject.GetComponent<WebCamTextureToMatHelper>();
        if (WebCamHelper != null)
        {
            WebCamHelper.outputColorFormat = inputGray ? WebCamTextureToMatHelper.ColorFormat.GRAY : WebCamTextureToMatHelper.ColorFormat.RGBA;
            Application.targetFrameRate = (int)WebCamHelper.requestedFPS;
            WebCamHelper.Initialize();
        }
        else throw new Exception("No webcamhelper!");
    }
    void FinishInitialize()
    {
        texture2 = new Texture2D(width, height, TextureFormat.RGBA32, false);
        texture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        imageOut.texture = texture;

        // set camera parameters.
        double fx;
        double fy;
        double cx;
        double cy;

        // string loadDirectoryPath = Path.Combine(Application.persistentDataPath, "ArUcoCameraCalibrationExample");
        // string calibratonDirectoryName = "camera_parameters1280x720";
        // string loadCalibratonFileDirectoryPath = Path.Combine(loadDirectoryPath, calibratonDirectoryName);
        // string loadPath = Path.Combine(loadCalibratonFileDirectoryPath, calibratonDirectoryName + ".xml");

        if (useStoredCameraParameters && xmlFile != null)
        {
            CameraParameters param;
            XmlSerializer serializer = new XmlSerializer(typeof(CameraParameters));
            using (var stream = new StringReader(xmlFile.text))
            {
                param = (CameraParameters)serializer.Deserialize(stream);
            }

            camMatrix = param.GetCameraMatrix();
            distCoeffs = new MatOfDouble(param.GetDistortionCoefficients());

            fx = param.camera_matrix[0];
            fy = param.camera_matrix[4];
            cx = param.camera_matrix[2];
            cy = param.camera_matrix[5];

            Debug.Log("Loaded CameraParameters from a stored XML file.");

        }
        else
        {
            int max_d = (int)Mathf.Max(width, height);
            fx = max_d;
            fy = max_d;
            cx = width / 2.0f;
            cy = height / 2.0f;

            camMatrix = new Mat(3, 3, CvType.CV_64FC1);
            camMatrix.put(0, 0, fx);
            camMatrix.put(0, 1, 0);
            camMatrix.put(0, 2, cx);
            camMatrix.put(1, 0, 0);
            camMatrix.put(1, 1, fy);
            camMatrix.put(1, 2, cy);
            camMatrix.put(2, 0, 0);
            camMatrix.put(2, 1, 0);
            camMatrix.put(2, 2, 1.0f);

            distCoeffs = new MatOfDouble(0, 0, 0, 0);

            Debug.Log("Created a dummy CameraParameters.");
        }

        Debug.Log("camMatrix " + camMatrix.dump());
        Debug.Log("distCoeffs " + distCoeffs.dump());
        // calibration camera matrix values.
        Size imageSize = new Size(width, height);
        double apertureWidth = 0;
        double apertureHeight = 0;
        double[] fovx = new double[1];
        double[] fovy = new double[1];
        double[] focalLength = new double[1];
        Point principalPoint = new Point(0, 0);
        double[] aspectratio = new double[1];

        Calib3d.calibrationMatrixValues(camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

        // To convert the difference of the FOV value of the OpenCV and Unity. 
        double fovXScale = (2.0 * Mathf.Atan((float)(imageSize.width / (2.0 * fx)))) / (Mathf.Atan2((float)cx, (float)fx) + Mathf.Atan2((float)(imageSize.width - cx), (float)fx));
        double fovYScale = (2.0 * Mathf.Atan((float)(imageSize.height / (2.0 * fy)))) / (Mathf.Atan2((float)cy, (float)fy) + Mathf.Atan2((float)(imageSize.height - cy), (float)fy));


        // Adjust Unity Camera FOV https://github.com/opencv/opencv/commit/8ed1945ccd52501f5ab22bdec6aa1f91f1e2cfd4

        // arCamera.fieldOfView = (float)(fovx[0] * fovXScale);
        arCamera.fieldOfView = (float)(fovy[0] * fovYScale);

        // Display objects near the camera.
        arCamera.nearClipPlane = 0.01f;



        //Detector parameters
        detectorParams = DetectorParameters.create();
        Debug.Log($"Default is {detectorParams.get_adaptiveThreshConstant()} " +
            $"max:{detectorParams.get_adaptiveThreshWinSizeMax()}" +
            $"min:{detectorParams.get_adaptiveThreshWinSizeMin()}" +
            $"step:{detectorParams.get_adaptiveThreshWinSizeStep()}" +
            $"corner max it:{detectorParams.get_cornerRefinementMaxIterations()}" + // 30
            $"corner ref meth:{detectorParams.get_cornerRefinementMethod()}" +      // 0 CORNER_REFINE_NONE
            $"corner ref min:{detectorParams.get_cornerRefinementMinAccuracy()}" +  // 0.1
            $"corner ref size:{detectorParams.get_cornerRefinementWinSize()}");     // 5

        detectorParams.set_adaptiveThreshWinSizeMax(40);
        detectorParams.set_minMarkerDistanceRate(0.02); //0.125

        // TODO : detectorParams.set // useAruco3Detection Better detection

        detectorParams.set_cornerRefinementMethod(Aruco.CORNER_REFINE_APRILTAG);
        // detectorParams.set_cornerRefinementMethod(1);
        // CORNER_REFINE_NONE = 0;
        // CORNER_REFINE_SUBPIX = 1;
        // CORNER_REFINE_CONTOUR = 2;
        // CORNER_REFINE_APRILTAG = 3;

        //detectorParams.set_adaptiveThreshWinSizeStep(4);
        //detectorParams.set_adaptiveThreshConstant(20);

        dictionary = Aruco.getPredefinedDictionary(ArUcoManager.Instance.TagType);

        smoothingPos = new Vector3[smoothingPoints];
        smoothingRot = new Quaternion[smoothingPoints];
    }

    public void OnWebCamTextureToMatHelperInitialized()
    {
        Debug.Log("OnWebCamTextureToMatHelperInitialized");

        Mat webCamTextureMat = WebCamHelper.GetMat();

        width = webCamTextureMat.width();
        height = webCamTextureMat.height();

        FinishInitialize();
    }

    public void OnWebCamTextureToMatHelperDisposed()
    {
        Debug.Log("OnWebCamTextureToMatHelperDisposed");

        if (texture != null)
        {
            Texture2D.Destroy(texture);
            texture = null;
        }

        //TODO : Dispose of stuff
    }

    public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
    {
        Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
    }

    void OnDestroy()
    {
        if (WebCamHelper != null)
            WebCamHelper.Dispose();
    }
    // Method to calculate the average of an array of Vector3
    public Vector3 CalculateAverage(Vector3[] vectors)
    {
        if (vectors == null || vectors.Length == 0)
        {
            Debug.LogError("The array is null or empty");
            return Vector3.zero; // Return zero vector if the array is null or empty
        }

        Vector3 sum = Vector3.zero;

        // Sum all vectors
        foreach (Vector3 vec in vectors)
        {
            sum += vec;
        }

        // Calculate the average
        Vector3 average = sum / vectors.Length;

        return average;
    }    
    
    // Method to calculate the average of an array of Quaternions
    public Quaternion CalculateAverage(Quaternion[] quaternions)
    {
        if (quaternions == null || quaternions.Length == 0)
        {
            Debug.LogError("The array is null or empty");
            return Quaternion.identity; // Return identity quaternion if the array is null or empty
        }

        // Initialize a matrix to accumulate the weighted sum of the outer products of the quaternions
        float[,] rotationMatrix = new float[4, 4];

        // Sum the outer products
        foreach (Quaternion q in quaternions)
        {
            float[] qArray = { q.x, q.y, q.z, q.w };
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    rotationMatrix[i, j] += qArray[i] * qArray[j];
                }
            }
        }

        // Normalize the rotation matrix
        float scale = 1.0f / quaternions.Length;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                rotationMatrix[i, j] *= scale;
            }
        }

        // Find the eigenvector of the largest eigenvalue of the rotation matrix
        Quaternion averageQuaternion = ExtractEigenvector(rotationMatrix);

        return averageQuaternion;
    }

    // Function to extract the eigenvector corresponding to the largest eigenvalue of a symmetric 4x4 matrix
    private Quaternion ExtractEigenvector(float[,] matrix)
    {
        // Use the power iteration method to find the dominant eigenvector
        Vector4 vec = new Vector4(1, 1, 1, 1).normalized;
        Vector4 vecNew = new Vector4();

        for (int iteration = 0; iteration < 10; iteration++)
        {
            vecNew.x = matrix[0, 0] * vec.x + matrix[0, 1] * vec.y + matrix[0, 2] * vec.z + matrix[0, 3] * vec.w;
            vecNew.y = matrix[1, 0] * vec.x + matrix[1, 1] * vec.y + matrix[1, 2] * vec.z + matrix[1, 3] * vec.w;
            vecNew.z = matrix[2, 0] * vec.x + matrix[2, 1] * vec.y + matrix[2, 2] * vec.z + matrix[2, 3] * vec.w;
            vecNew.w = matrix[3, 0] * vec.x + matrix[3, 1] * vec.y + matrix[3, 2] * vec.z + matrix[3, 3] * vec.w;
            vecNew.Normalize();
            vec = vecNew;
        }

        return new Quaternion(vec.x, vec.y, vec.z, vec.w);
    }
}
