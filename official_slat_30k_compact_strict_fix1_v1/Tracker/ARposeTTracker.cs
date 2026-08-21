using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using Unity.Collections;
using UnityEngine.EventSystems;
using System.Globalization;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.Rendering;
using System.IO;
using System.Text;

[System.Serializable]
public class SelectionData
{
    public List<int> selected; 
}

[System.Serializable]
public class PreprocessResponse
{
    public string status;
    public int total_images;
    public string session_id;
}

[System.Serializable]
public class ServerStatusResponse
{
    public string status;
    public string message;
}

[System.Serializable]
public class MobileARResponse
{
    public string format;
    public string mesh_url;
    public string coordinate_frame;
    public string placement;
}

[System.Serializable]
public class GenerateResponse
{
    public string status;
    public string message;
    public string session_id;
    public MobileARResponse mobile_ar;
}

[System.Serializable]
public class SlamPointData
{
    public float x;
    public float y;
    public float z;
    public float confidence;
}

[System.Serializable]
public class SlamPointPayload
{
    public string schema = "arpose_tracker_frame_points_v1";
    public int point_count;
    public List<SlamPointData> points = new List<SlamPointData>();
}

public class SegClick
{
    public float x;
    public float y;
    public int label;

    public SegClick(float xNorm, float yNorm, int pointLabel)
    {
        x = xNorm;
        y = yNorm;
        label = pointLabel;
    }
}

public class ARPoseTracker : MonoBehaviour
{
    [Header("AR 核心组件")]
    public Transform arCamera;
    public ARSession arSession;
    public ARCameraManager cameraManager;
    public ARPointCloudManager pointCloudManager;
    public ARAnchorManager anchorManager;

    [Header("AR/SLAM 点云上传")]
    public bool uploadSlamPoints = true;
    public int maxSlamPointsPerFrame = 800;
    public int slamPointStride = 1;

    [Header("主 UI")]
    public Text debugText;
    public Button recordButton;
    public Text buttonText;

    [Header("竖屏状态文字")]
    public bool fitStatusTextToPortraitSafeArea = true;
    [Range(0.15f, 0.60f)] public float statusTextHeightRatio = 0.30f;
    public int statusTextMinFontSize = 12;
    public int statusTextMaxFontSize = 26;
    public int statusTextMaxCharacters = 280;

    [Header("人工审核 UI")]
    public GameObject reviewPanel;       
    public RawImage previewImage;        
    public Text imageIndexText;          
    public Text keepStatusText;          
    public Button btnPrev;               
    public Button btnNext;               
    public Button btnToggleKeep;         
    public Button btnSubmit;
    public Button btnCancel;             // 新增：退出/重拍按钮

    [Header("网络设置")]
    public string serverURL = "http://10.102.33.100:5000";
    public float sendInterval = 0.5f;
    public float maxCameraFrameTimestampDeltaSeconds = 0.05f;

    [Header("AR 轨迹稳定性")]
    public float trackingWarmupSeconds = 1.5f;
    public float trackingResetTimeoutSeconds = 15.0f;
    public float maxConsecutiveCameraPoseJumpMeters = 0.75f;
    public float maxConsecutiveCameraPoseJumpDegrees = 45.0f;

    [Header("姿态分散实时采样")]
    [Tooltip("先按固定物体中心估计做在线角度去重；服务端仍会用分割后物体中心执行最终8视角球面最远点选择")]
    public bool enablePoseDiverseCapture = true;
    [Tooltip("录制开始时沿首帧相机前向估计物体中心的距离（米）")]
    public float assumedObjectDistanceMeters = 0.70f;
    [Range(1.0f, 45.0f)]
    public float minimumPoseDiversityAngleDegrees = 10.0f;
    [Tooltip("UI建议的候选帧数；最终模型输入仍由服务端筛成8视角")]
    public int recommendedPoseDiverseFrameCount = 24;

    [Header("重建 Mesh AR 显示")]
    public Button meshDisplayButton;
    public Text meshDisplayModeText;
    [Tooltip("请在 Unity Inspector 中绑定使用 URP/Unlit 的透明材质，避免 Android 构建裁剪 Shader")]
    public Material reconstructedSurfaceMaterialTemplate;
    [Tooltip("请在 Unity Inspector 中绑定使用 URP/Unlit 的不透明材质，避免 Android 构建裁剪 Shader")]
    public Material reconstructedOutlineMaterialTemplate;
    public Color reconstructedSurfaceColor = new Color(0.10f, 0.85f, 0.55f, 0.42f);
    public Color reconstructedOutlineColor = new Color(0.05f, 1.0f, 0.70f, 1.0f);
    [Tooltip("Anchor 恢复 Tracking 后持续稳定多久才重新显示 Mesh")]
    public float reconstructedAnchorRecoveryStableSeconds = 0.75f;
    public int maxMobileMeshVertices = 200000;
    public int maxMobileMeshIndices = 600000;

    private float timer = 0f;
    private bool isSending = false;
    private bool isRecording = false;
    private bool isPreparingRecording = false;
    private bool captureWorldFrameValid = false;
    private bool hasPreviousCaptureWorldPose = false;
    private Vector3 previousCaptureWorldCameraPosition;
    private Quaternion previousCaptureWorldCameraRotation;
    private Texture2D cameraTexture;

    private GameObject reconstructedMeshRoot;
    private GameObject reconstructedSurfaceObject;
    private GameObject reconstructedOutlineObject;
    private Mesh reconstructedSurfaceMesh;
    private Mesh reconstructedOutlineMesh;
    private Material reconstructedSurfaceMaterial;
    private Material reconstructedOutlineMaterial;
    private int reconstructedMeshDisplayMode = 0;
    private GameObject captureReferenceAnchorObject;
    private ARAnchor captureReferenceAnchor;
    private Vector3 captureReferenceAnchorPosition;
    private Quaternion captureReferenceAnchorRotation = Quaternion.identity;
    private bool captureReferenceAnchorPoseValid = false;
    private bool captureReferenceAnchorDisplayStable = false;
    private bool captureReferenceAnchorEverTracked = false;
    private float captureReferenceAnchorTrackingSince = -1.0f;
    private TrackingState lastCaptureReferenceAnchorTrackingState = TrackingState.None;
    private Rect lastStatusSafeArea = new Rect(-1.0f, -1.0f, -1.0f, -1.0f);

    private List<Texture2D> downloadedPreviews = new List<Texture2D>();
    private List<List<SegClick>> frameClicks = new List<List<SegClick>>();
    private HashSet<int> approvedSeedFrames = new HashSet<int>();
    private int currentPreviewIndex = 0;
    private int capturedFrameCount = 0;
    private bool addForegroundPoint = true;
    private bool segmentationReady = false;
    private Texture2D currentDisplayTexture;
    private AspectRatioFitter previewAspectFitter;
    private const string CpuImageTransformName = "None";
    private bool hasCameraFrameSnapshot = false;
    private Vector3 cameraFramePosition;
    private Quaternion cameraFrameRotation;
    private bool cameraFrameHasIntrinsics = false;
    private XRCameraIntrinsics cameraFrameIntrinsics;
    private long cameraFrameTimestampNs = -1;
    private double cameraFramePoseSampleSeconds = -1.0;
    private Matrix4x4 cameraFrameDisplayMatrix = Matrix4x4.identity;
    private Matrix4x4 cameraFrameProjectionMatrix = Matrix4x4.identity;
    private bool poseDiversityTargetValid = false;
    private Vector3 poseDiversityTargetPosition;
    private readonly List<Vector3> acceptedPoseDiversityDirections = new List<Vector3>();
    private float lastPoseDiversityMinimumAngle = -1.0f;

    void Start()
    {
        if (anchorManager == null)
            anchorManager = FindObjectOfType<ARAnchorManager>();
        if (cameraManager != null) cameraManager.frameReceived += OnCameraFrameReceived;
        if (reviewPanel != null) reviewPanel.SetActive(false);
        recordButton.onClick.AddListener(ToggleRecording);
        buttonText.text = "开始录制";
        ConfigureStatusTextForPortrait();

        btnPrev.onClick.AddListener(() => ChangePreview(-1));
        btnNext.onClick.AddListener(() => ChangePreview(1));
        btnToggleKeep.onClick.AddListener(TogglePointMode);
        btnSubmit.onClick.AddListener(SubmitSelection);
        
        // 绑定退出按钮
        if (btnCancel != null) btnCancel.onClick.AddListener(CancelReview);
        if (meshDisplayButton != null)
        {
            meshDisplayButton.onClick.AddListener(ToggleReconstructedMeshDisplay);
            meshDisplayButton.gameObject.SetActive(false);
        }

        if (previewImage != null)
        {
            previewAspectFitter = previewImage.GetComponent<AspectRatioFitter>();
            if (previewAspectFitter == null) previewAspectFitter = previewImage.gameObject.AddComponent<AspectRatioFitter>();
            previewAspectFitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
            previewImage.raycastTarget = true;

            EventTrigger trigger = previewImage.gameObject.GetComponent<EventTrigger>();
            if (trigger == null) trigger = previewImage.gameObject.AddComponent<EventTrigger>();
            EventTrigger.Entry entry = new EventTrigger.Entry();
            entry.eventID = EventTriggerType.PointerClick;
            entry.callback.AddListener((eventData) => OnPreviewClicked((PointerEventData)eventData));
            trigger.triggers.Add(entry);
        }
    }

    void OnDestroy()
    {
        if (cameraManager != null) cameraManager.frameReceived -= OnCameraFrameReceived;
        ClearReconstructedMesh();
        ClearCaptureReferenceAnchor();
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (arCamera == null) return;
        Vector3 currentPosition = arCamera.position;
        Quaternion currentRotation = arCamera.rotation;
        if (isRecording && captureWorldFrameValid && hasPreviousCaptureWorldPose)
        {
            float poseStep = Vector3.Distance(
                previousCaptureWorldCameraPosition, currentPosition
            );
            float rotationStep = Quaternion.Angle(
                previousCaptureWorldCameraRotation, currentRotation
            );
            if (
                (maxConsecutiveCameraPoseJumpMeters > 0.0f
                    && poseStep > maxConsecutiveCameraPoseJumpMeters)
                || (maxConsecutiveCameraPoseJumpDegrees > 0.0f
                    && rotationStep > maxConsecutiveCameraPoseJumpDegrees)
            )
            {
                isRecording = false;
                captureWorldFrameValid = false;
                hasPreviousCaptureWorldPose = false;
                ClearReconstructedMesh();
                StartCoroutine(AbortRecordingAfterPoseJump(poseStep, rotationStep));
            }
        }
        if (isRecording && captureWorldFrameValid)
        {
            previousCaptureWorldCameraPosition = currentPosition;
            previousCaptureWorldCameraRotation = currentRotation;
            hasPreviousCaptureWorldPose = true;
        }
        cameraFramePosition = currentPosition;
        cameraFrameRotation = currentRotation;
        cameraFramePoseSampleSeconds = (double)Time.realtimeSinceStartup;
        cameraFrameHasIntrinsics =
            cameraManager != null && cameraManager.TryGetIntrinsics(out cameraFrameIntrinsics);
        cameraFrameTimestampNs = args.timestampNs.HasValue ? args.timestampNs.Value : -1;
        cameraFrameDisplayMatrix =
            args.displayMatrix.HasValue ? args.displayMatrix.Value : Matrix4x4.identity;
        cameraFrameProjectionMatrix =
            args.projectionMatrix.HasValue ? args.projectionMatrix.Value : Matrix4x4.identity;
        hasCameraFrameSnapshot = true;
    }

    void ToggleRecording()
    {
        if (isPreparingRecording) return;
        if (!isRecording)
        {
            StartCoroutine(BeginRecordingWithFreshARSession());
        }
        else
        {
            isRecording = false;
            bool anchorCreated = CreateCaptureReferenceAnchor();
            recordButton.gameObject.SetActive(false); 
            UpdateUI(
                anchorCreated
                    ? "采集结束，AR 锚点已建立，正在准备原图点选..."
                    : "采集结束，但 AR 锚点建立失败；Mesh 将无法可靠叠加",
                anchorCreated ? Color.yellow : Color.red
            );
            StartCoroutine(RequestPreprocess());
        }
    }

    IEnumerator BeginRecordingWithFreshARSession()
    {
        isPreparingRecording = true;
        isRecording = false;
        captureWorldFrameValid = false;
        hasPreviousCaptureWorldPose = false;
        ResetPoseDiverseCapture();
        ClearReconstructedMesh();
        ClearCaptureReferenceAnchor();
        hasCameraFrameSnapshot = false;
        timer = 0f;
        segmentationReady = false;
        addForegroundPoint = true;
        approvedSeedFrames.Clear();
        recordButton.interactable = false;
        buttonText.text = "正在重置并稳定 AR...";
        buttonText.color = Color.yellow;

        if (arSession != null) arSession.Reset();
        float deadline = Time.realtimeSinceStartup + Mathf.Max(1.0f, trackingResetTimeoutSeconds);
        float stableSince = -1.0f;
        while (Time.realtimeSinceStartup < deadline)
        {
            bool stable =
                ARSession.state == ARSessionState.SessionTracking
                && hasCameraFrameSnapshot;
            if (stable)
            {
                if (stableSince < 0.0f) stableSince = Time.realtimeSinceStartup;
                if (
                    Time.realtimeSinceStartup - stableSince
                    >= Mathf.Max(0.0f, trackingWarmupSeconds)
                )
                    break;
            }
            else
            {
                stableSince = -1.0f;
            }
            yield return null;
        }
        if (
            stableSince < 0.0f
            || Time.realtimeSinceStartup - stableSince
                < Mathf.Max(0.0f, trackingWarmupSeconds)
        )
        {
            isPreparingRecording = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI("AR 跟踪未在限定时间内稳定，请调整环境后重试", Color.red);
            yield break;
        }

        yield return StartCoroutine(SendCommand("/start_record"));
        InitializePoseDiversityTarget(cameraFramePosition, cameraFrameRotation);
        captureWorldFrameValid = true;
        previousCaptureWorldCameraPosition = cameraFramePosition;
        previousCaptureWorldCameraRotation = cameraFrameRotation;
        hasPreviousCaptureWorldPose = true;
        isRecording = true;
        isPreparingRecording = false;
        recordButton.interactable = true;
        buttonText.text = "结束录制并分割";
        buttonText.color = Color.red;
        UpdateUI("AR 已重置并稳定，开始采集", Color.green);
    }

    IEnumerator AbortRecordingAfterPoseJump(
        float translationDeltaMeters,
        float rotationDeltaDegrees
    )
    {
        isPreparingRecording = true;
        recordButton.interactable = false;
        UpdateUI(
            $"检测到 AR 世界坐标跳变 {translationDeltaMeters:F2} m / "
            + $"{rotationDeltaDegrees:F1}°，本次采集已取消，请重拍",
            Color.red
        );
        yield return StartCoroutine(SendCommand("/cancel_review"));
        ResetPoseDiverseCapture();
        isSending = false;
        isPreparingRecording = false;
        recordButton.gameObject.SetActive(true);
        recordButton.interactable = true;
        buttonText.text = "重新录制";
        buttonText.color = Color.white;
    }

    // ========== 更新：全局取消与重置 ==========
    void CancelReview()
    {
        // 1. 强行打断所有的前端状态
        isRecording = false;
        isPreparingRecording = false;
        captureWorldFrameValid = false;
        hasPreviousCaptureWorldPose = false;
        ResetPoseDiverseCapture();
        isSending = false;
        segmentationReady = false;
        addForegroundPoint = true;
        frameClicks.Clear();
        approvedSeedFrames.Clear();
        ClearReconstructedMesh();
        ClearCaptureReferenceAnchor();
        
        // 2. 隐藏审核面板（如果它正开着的话）
        if (reviewPanel != null) reviewPanel.SetActive(false);
        
        // 3. 恢复主界面的录制按钮
        recordButton.gameObject.SetActive(true);
        buttonText.text = "开始录制";
        buttonText.color = Color.white;
        
        UpdateUI("已强制重置，可随时开始新的录制", Color.white);

        // 4. 通知服务器中断当前任务并清空缓存
        StartCoroutine(SendCommand("/cancel_review"));
    }
    // ==========================================

    void Update()
    {
        if (fitStatusTextToPortraitSafeArea && Screen.safeArea != lastStatusSafeArea)
            ConfigureStatusTextForPortrait();
        UpdateCaptureReferenceAnchorTracking();
        if (ARSession.state != ARSessionState.SessionTracking) return;

        if (arCamera != null && isRecording)
        {
            Vector3 pos = arCamera.position;
            Quaternion quat = arCamera.rotation;
            string nearestPoseAngleText = lastPoseDiversityMinimumAngle < 0.0f
                ? "首帧"
                : lastPoseDiversityMinimumAngle.ToString("F1") + "°";
            string diversityStatus = enablePoseDiverseCapture
                ? $"姿态分散帧: {acceptedPoseDiversityDirections.Count}/"
                    + $"{Mathf.Max(8, recommendedPoseDiverseFrameCount)}; "
                    + "最近角距: " + nearestPoseAngleText
                : "姿态分散筛选: 关闭";
            UpdateUI(
                $"[录制中] {diversityStatus}\nPos: {pos.x:F2}, {pos.y:F2}, {pos.z:F2}",
                Color.green
            );

            timer += Time.deltaTime;
            if (timer >= sendInterval && !isSending)
            {
                timer = 0f;
                Vector3 framePosition = hasCameraFrameSnapshot ? cameraFramePosition : pos;
                Quaternion frameRotation = hasCameraFrameSnapshot ? cameraFrameRotation : quat;
                if (
                    !EvaluatePoseDiverseCandidate(
                        framePosition,
                        frameRotation,
                        out Vector3 diversityDirection,
                        out float diversityMinimumAngle
                    )
                )
                {
                    UpdateUI(
                        $"视角与已有帧过近 ({diversityMinimumAngle:F1}° < "
                        + $"{minimumPoseDiversityAngleDegrees:F1}°)，请绕物体移动",
                        Color.yellow
                    );
                    return;
                }
                if (cameraManager != null && cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
                {
                    XRCameraIntrinsics intrinsics = cameraFrameIntrinsics;
                    bool hasIntrinsics = cameraFrameHasIntrinsics;
                    long frameTimestampNs = hasCameraFrameSnapshot ? cameraFrameTimestampNs : -1;
                    double cpuTimestampSeconds = image.timestamp;
                    double frameTimestampSeconds =
                        frameTimestampNs > 0 ? frameTimestampNs * 1.0e-9 : -1.0;
                    double timestampDeltaSeconds =
                        frameTimestampSeconds > 0.0 && cpuTimestampSeconds > 0.0
                            ? System.Math.Abs(cpuTimestampSeconds - frameTimestampSeconds)
                            : -1.0;
                    if (
                        maxCameraFrameTimestampDeltaSeconds > 0.0f
                        && timestampDeltaSeconds >= 0.0
                        && timestampDeltaSeconds > maxCameraFrameTimestampDeltaSeconds
                    )
                    {
                        image.Dispose();
                        UpdateUI(
                            $"相机帧不同步，跳过本帧 ({timestampDeltaSeconds * 1000.0:F1} ms)",
                            Color.yellow
                        );
                        return;
                    }
                    StartCoroutine(
                        ProcessAndSendData(
                            image,
                            framePosition,
                            frameRotation,
                            hasIntrinsics,
                            intrinsics,
                            cpuTimestampSeconds,
                            frameTimestampNs,
                            hasCameraFrameSnapshot ? cameraFramePoseSampleSeconds : (double)Time.realtimeSinceStartup,
                            timestampDeltaSeconds,
                            hasCameraFrameSnapshot ? cameraFrameDisplayMatrix : Matrix4x4.identity,
                            hasCameraFrameSnapshot ? cameraFrameProjectionMatrix : Matrix4x4.identity,
                            hasCameraFrameSnapshot ? "camera_frame_received" : "legacy_update_pose",
                            diversityDirection,
                            diversityMinimumAngle
                        )
                    );
                }
            }
        }
    }

    IEnumerator RequestPreprocess()
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + "/preprocess", ""))
        {
            www.timeout = 120; 
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                PreprocessResponse response = JsonUtility.FromJson<PreprocessResponse>(www.downloadHandler.text);
                int totalImages = response.total_images;
                capturedFrameCount = totalImages;
                
                UpdateUI($"采集完成，共 {totalImages} 张。请在原图上点击物体前景/背景点进行监督分割。", Color.green);
                StartCoroutine(DownloadAllFrames(totalImages));
            }
            else
            {
                UpdateUI("准备分割失败: " + www.error, Color.red);
                // 失败时也把重拍按钮亮出来
                recordButton.gameObject.SetActive(true);
                buttonText.text = "开始录制";
                buttonText.color = Color.white;
            }
        }
    }

    IEnumerator DownloadAllFrames(int count)
    {
        downloadedPreviews.Clear();
        frameClicks.Clear();
        approvedSeedFrames.Clear();
        for (int i = 0; i < count; i++)
        {
            UpdateUI($"正在下载原图 {i + 1}/{count}...", Color.white);
            using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_frame/{i}"))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    downloadedPreviews.Add(DownloadHandlerTexture.GetContent(www));
                    frameClicks.Add(new List<SegClick>());
                }
            }
        }

        reviewPanel.SetActive(true);
        currentPreviewIndex = 0;
        RefreshReviewUI();
    }

    void ChangePreview(int step)
    {
        currentPreviewIndex += step;
        if (currentPreviewIndex < 0) currentPreviewIndex = downloadedPreviews.Count - 1;
        if (currentPreviewIndex >= downloadedPreviews.Count) currentPreviewIndex = 0;
        RefreshReviewUI();
    }

    void TogglePointMode()
    {
        addForegroundPoint = !addForegroundPoint;
        RefreshReviewUI();
    }

    void SetButtonLabel(Button button, string label)
    {
        if (button == null) return;
        Text text = button.GetComponentInChildren<Text>();
        if (text != null) text.text = label;
    }

    void RefreshReviewUI()
    {
        if (downloadedPreviews.Count == 0) return;
        Texture2D baseTexture = downloadedPreviews[currentPreviewIndex];
        if (previewAspectFitter != null && baseTexture.height > 0)
            previewAspectFitter.aspectRatio = (float)baseTexture.width / baseTexture.height;

        if (currentDisplayTexture != null && !downloadedPreviews.Contains(currentDisplayTexture))
            Destroy(currentDisplayTexture);
        currentDisplayTexture = segmentationReady ? baseTexture : BuildTextureWithClicks(baseTexture, currentPreviewIndex);
        previewImage.texture = currentDisplayTexture;
        imageIndexText.text = $"当前: {currentPreviewIndex + 1} / {downloadedPreviews.Count}";

        bool frameApproved = approvedSeedFrames.Contains(currentPreviewIndex);
        bool frameHasClicks = FrameHasLocalClicks(currentPreviewIndex);

        if (segmentationReady)
        {
            keepStatusText.text = "分割预览: 满意请提交；不满意继续点前景/背景";
            keepStatusText.color = Color.green;
            SetButtonLabel(btnSubmit, "确认生成");
        }
        else if (frameApproved)
        {
            keepStatusText.text = $"当前帧已作为种子；种子帧 {approvedSeedFrames.Count} 张";
            keepStatusText.color = Color.green;
            SetButtonLabel(btnSubmit, "运行分割");
        }
        else if (frameHasClicks)
        {
            keepStatusText.text = addForegroundPoint ? "检查当前预览，满意请确认本帧为种子" : "检查当前预览，满意请确认本帧为种子";
            keepStatusText.color = Color.yellow;
            SetButtonLabel(btnSubmit, "确认本帧");
        }
        else
        {
            keepStatusText.text = approvedSeedFrames.Count > 0
                ? $"可运行分割；也可继续点选更多种子帧，当前种子 {approvedSeedFrames.Count} 张"
                : (addForegroundPoint ? "点选模式: 前景点，先确认至少一帧种子" : "点选模式: 背景点，先确认至少一帧种子");
            keepStatusText.color = addForegroundPoint ? Color.green : Color.red;
            SetButtonLabel(btnSubmit, approvedSeedFrames.Count > 0 ? "运行分割" : "确认本帧");
        }
        SetButtonLabel(btnToggleKeep, addForegroundPoint ? "前景点" : "背景点");
    }

    bool FrameHasLocalClicks(int frameIndex)
    {
        return frameIndex >= 0 && frameIndex < frameClicks.Count && frameClicks[frameIndex].Count > 0;
    }

    void OnPreviewClicked(PointerEventData eventData)
    {
        if (reviewPanel == null || !reviewPanel.activeSelf || previewImage == null) return;

        RectTransform rect = previewImage.rectTransform;
        Vector2 localPoint;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(rect, eventData.position, eventData.pressEventCamera, out localPoint))
            return;

        Rect imageRect = GetDisplayedImageRect();
        if (!imageRect.Contains(localPoint)) return;

        float xNorm = Mathf.Clamp01((localPoint.x - imageRect.xMin) / imageRect.width);
        float yNorm = Mathf.Clamp01(1.0f - ((localPoint.y - imageRect.yMin) / imageRect.height));
        int label = addForegroundPoint ? 1 : 0;
        if (segmentationReady)
            segmentationReady = false;
        approvedSeedFrames.Remove(currentPreviewIndex);
        frameClicks[currentPreviewIndex].Add(new SegClick(xNorm, yNorm, label));
        RefreshReviewUI();
        StartCoroutine(SendSegPoint(currentPreviewIndex, xNorm, yNorm, label));
    }

    Rect GetDisplayedImageRect()
    {
        Rect r = previewImage.rectTransform.rect;
        Texture tex = previewImage.texture;
        if (tex == null || tex.height <= 0) return r;

        float imageAspect = (float)tex.width / tex.height;
        float rectAspect = r.width / r.height;
        if (rectAspect > imageAspect)
        {
            float width = r.height * imageAspect;
            float x = r.xMin + (r.width - width) * 0.5f;
            return new Rect(x, r.yMin, width, r.height);
        }
        else
        {
            float height = r.width / imageAspect;
            float y = r.yMin + (r.height - height) * 0.5f;
            return new Rect(r.xMin, y, r.width, height);
        }
    }

    Texture2D BuildTextureWithClicks(Texture2D source, int frameIndex)
    {
        Texture2D tex = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
        tex.SetPixels32(source.GetPixels32());

        if (frameIndex >= 0 && frameIndex < frameClicks.Count)
        {
            foreach (SegClick click in frameClicks[frameIndex])
            {
                int x = Mathf.RoundToInt(click.x * (source.width - 1));
                int y = Mathf.RoundToInt((1.0f - click.y) * (source.height - 1));
                DrawClickMarker(tex, x, y, click.label == 1 ? Color.green : Color.red);
            }
        }

        tex.Apply();
        return tex;
    }

    void DrawClickMarker(Texture2D tex, int cx, int cy, Color color)
    {
        int radius = Mathf.Max(6, Mathf.RoundToInt(Mathf.Min(tex.width, tex.height) * 0.012f));
        Color outline = Color.white;
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int x = cx + dx;
                int y = cy + dy;
                if (x < 0 || y < 0 || x >= tex.width || y >= tex.height) continue;
                float d = Mathf.Sqrt(dx * dx + dy * dy);
                if (d <= radius)
                    tex.SetPixel(x, y, color);
                if (Mathf.Abs(d - radius) < 1.8f)
                    tex.SetPixel(x, y, outline);
            }
        }
    }

    IEnumerator SendSegPoint(int frameIndex, float xNorm, float yNorm, int label)
    {
        string json = string.Format(
            CultureInfo.InvariantCulture,
            "{{\"frame_index\":{0},\"x\":{1:F6},\"y\":{2:F6},\"label\":{3},\"normalized\":true}}",
            frameIndex,
            xNorm,
            yNorm,
            label
        );
        UnityWebRequest www = new UnityWebRequest(serverURL + "/add_seg_point", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
        {
            UpdateUI($"已添加{(label == 1 ? "前景" : "背景")}点: frame {frameIndex}, ({xNorm:F2},{yNorm:F2})", Color.green);
            StartCoroutine(UpdateCurrentFrameSegmentationPreview(frameIndex));
        }
        else
            UpdateUI("添加分割点失败: " + www.error, Color.red);
    }

    IEnumerator UpdateCurrentFrameSegmentationPreview(int frameIndex)
    {
        using (UnityWebRequest seg = UnityWebRequest.PostWwwForm(serverURL + $"/segment_frame/{frameIndex}", ""))
        {
            seg.timeout = 120;
            yield return seg.SendWebRequest();
            if (seg.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("当前帧分割失败: " + seg.error, Color.red);
                yield break;
            }
        }

        using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_prompt_preview/{frameIndex}?t={Time.realtimeSinceStartup}"))
        {
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success && frameIndex >= 0 && frameIndex < downloadedPreviews.Count)
            {
                downloadedPreviews[frameIndex] = DownloadHandlerTexture.GetContent(www);
                if (frameIndex == currentPreviewIndex)
                    RefreshReviewUI();
            }
        }
    }

    void SubmitSelection()
    {
        if (!segmentationReady)
        {
            if (!approvedSeedFrames.Contains(currentPreviewIndex) && FrameHasLocalClicks(currentPreviewIndex))
            {
                UpdateUI("正在确认当前帧为视频传播种子...", Color.yellow);
                StartCoroutine(ApproveCurrentSeedFrame(currentPreviewIndex));
                return;
            }

            if (approvedSeedFrames.Count == 0)
            {
                UpdateUI("请先在至少一帧上点选，并确认该帧为种子。", Color.red);
                return;
            }

            UpdateUI($"正在使用 {approvedSeedFrames.Count} 张种子帧进行整段视频分割...", Color.yellow);
            StartCoroutine(RunSegmentationAndPreview());
            return;
        }

        reviewPanel.SetActive(false);
        UpdateUI("分割已确认，正在检查采集视角覆盖...", Color.yellow);
        StartCoroutine(CheckInputQcThenGenerate());
    }

    IEnumerator ApproveCurrentSeedFrame(int frameIndex)
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + $"/approve_seed_frame/{frameIndex}", ""))
        {
            www.timeout = 120;
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success)
            {
                approvedSeedFrames.Add(frameIndex);
                UpdateUI($"已确认第 {frameIndex + 1} 帧为种子。可继续确认其它帧，或再次提交运行分割。", Color.green);
                RefreshReviewUI();
            }
            else
            {
                UpdateUI("确认种子帧失败: " + www.error, Color.red);
            }
        }
    }

    IEnumerator RunSegmentationAndPreview()
    {
        using (UnityWebRequest seg = UnityWebRequest.PostWwwForm(serverURL + "/run_segmentation", ""))
        {
            seg.timeout = 600;
            yield return seg.SendWebRequest();
            if (seg.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("视频分割失败: " + seg.error, Color.red);
                recordButton.gameObject.SetActive(true);
                buttonText.text = "开始录制";
                buttonText.color = Color.white;
                yield break;
            }
        }

        yield return StartCoroutine(DownloadSegmentedPreviews(downloadedPreviews.Count));
        segmentationReady = true;
        currentPreviewIndex = 0;
        reviewPanel.SetActive(true);
        RefreshReviewUI();
        UpdateUI("分割完成，请检查 preview，确认后再次提交生成。", Color.green);
    }

    IEnumerator DownloadSegmentedPreviews(int count)
    {
        downloadedPreviews.Clear();
        frameClicks.Clear();
        for (int i = 0; i < count; i++)
        {
            UpdateUI($"正在下载分割 preview {i + 1}/{count}...", Color.white);
            using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_preview/{i}"))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    downloadedPreviews.Add(DownloadHandlerTexture.GetContent(www));
                    frameClicks.Add(new List<SegClick>());
                }
            }
        }
    }

    IEnumerator SendGenerateFromAllFrames()
    {
        SelectionData data = new SelectionData();
        data.selected = new List<int>();
        for (int i = 0; i < capturedFrameCount; i++)
        {
            data.selected.Add(i);
        }

        string jsonPayload = JsonUtility.ToJson(data);
        yield return StartCoroutine(SendGenerateCommand(jsonPayload));
    }

    IEnumerator CheckInputQcThenGenerate()
    {
        SelectionData data = new SelectionData();
        data.selected = new List<int>();
        for (int i = 0; i < capturedFrameCount; i++)
        {
            data.selected.Add(i);
        }

        string jsonPayload = JsonUtility.ToJson(data);
        UnityWebRequest qc = new UnityWebRequest(serverURL + "/input_qc", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        qc.uploadHandler = new UploadHandlerRaw(bodyRaw);
        qc.downloadHandler = new DownloadHandlerBuffer();
        qc.SetRequestHeader("Content-Type", "application/json");
        qc.timeout = 120;
        yield return qc.SendWebRequest();

        if (qc.result != UnityWebRequest.Result.Success)
        {
            UpdateUI("输入检查失败: " + ExtractServerMessage(qc), Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
            yield break;
        }

        string responseText = qc.downloadHandler.text;
        ServerStatusResponse response = JsonUtility.FromJson<ServerStatusResponse>(responseText);
        if (response != null && response.status == "warning")
        {
            string msg = string.IsNullOrEmpty(response.message) ? "输入视角覆盖不足，请补采或重拍。" : response.message;
            UpdateUI(msg, Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "重新录制";
            buttonText.color = Color.white;
            yield break;
        }

        UpdateUI("输入检查通过，服务器正在重建 Mesh (需耐心等待)！", Color.yellow);
        yield return StartCoroutine(SendGenerateCommand(jsonPayload));
    }

    IEnumerator SendGenerateCommand(string json)
    {
        UnityWebRequest www = new UnityWebRequest(serverURL + "/generate", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        
        www.timeout = 600; 

        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
        {
            GenerateResponse response = null;
            try
            {
                response = JsonUtility.FromJson<GenerateResponse>(www.downloadHandler.text);
            }
            catch
            {
                response = null;
            }
            string msg = response != null ? response.message : ExtractServerMessage(www);
            bool hasMobileMesh =
                response != null
                && response.mobile_ar != null
                && !string.IsNullOrEmpty(response.mobile_ar.mesh_url);
            if (
                hasMobileMesh
                && HasCaptureReferenceAnchor()
            )
            {
                UpdateUI("Mesh 重建完成，正在加载到原物体位置...", Color.yellow);
                yield return StartCoroutine(
                    DownloadAndDisplayReconstructedMesh(response.mobile_ar.mesh_url)
                );
            }
            else if (hasMobileMesh && !HasCaptureReferenceAnchor())
            {
                UpdateUI(
                    "Mesh 已重建，但采集 Anchor 已丢失；为避免错位未叠加，请重新采集",
                    Color.red
                );
            }
            else
            {
                UpdateUI(
                    string.IsNullOrEmpty(msg) ? "重建完成，但服务端未返回 AR Mesh" : msg,
                    Color.green
                );
            }
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
        }
        else
        {
            UpdateUI("生成失败: " + ExtractServerMessage(www), Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
        }
    }

    IEnumerator DownloadAndDisplayReconstructedMesh(string meshUrl)
    {
        string url = meshUrl.StartsWith("http://") || meshUrl.StartsWith("https://")
            ? meshUrl
            : serverURL.TrimEnd('/') + "/" + meshUrl.TrimStart('/');
        using (UnityWebRequest meshRequest = UnityWebRequest.Get(url))
        {
            meshRequest.timeout = 180;
            yield return meshRequest.SendWebRequest();
            if (meshRequest.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("AR Mesh 下载失败: " + ExtractServerMessage(meshRequest), Color.red);
                yield break;
            }
            if (!HasCaptureReferenceAnchor())
            {
                UpdateUI("采集 Anchor 已丢失，已取消 Mesh 叠加", Color.red);
                yield break;
            }

            Vector3[] vertices;
            Vector3[] normals;
            int[] triangles;
            try
            {
                ParseMobileARMesh(
                    meshRequest.downloadHandler.data,
                    out vertices,
                    out normals,
                    out triangles
                );
            }
            catch (System.Exception exception)
            {
                ClearReconstructedMesh();
                UpdateUI("AR Mesh 解析失败: " + exception.Message, Color.red);
                yield break;
            }

            try
            {
                CreateReconstructedMeshOverlay(vertices, normals, triangles);
                if (captureReferenceAnchorDisplayStable)
                {
                    UpdateUI(
                        $"重建完成：Mesh 已固定在原物体位置 "
                        + $"({vertices.Length} 顶点 / {triangles.Length / 3} 三角形)",
                        Color.green
                    );
                }
                else
                {
                    UpdateUI(
                        "Mesh 已加载，等待 AR Anchor 恢复稳定后自动显示",
                        Color.yellow
                    );
                }
            }
            catch (System.Exception exception)
            {
                ClearReconstructedMesh();
                UpdateUI("AR Mesh 材质/显示失败: " + exception.Message, Color.red);
            }
        }
    }

    void ParseMobileARMesh(
        byte[] payload,
        out Vector3[] vertices,
        out Vector3[] normals,
        out int[] triangles
    )
    {
        vertices = null;
        normals = null;
        triangles = null;
        if (payload == null || payload.Length < 24)
            throw new InvalidDataException("Mesh 数据不完整");

        using (MemoryStream stream = new MemoryStream(payload, false))
        using (BinaryReader reader = new BinaryReader(stream))
        {
            string magic = Encoding.ASCII.GetString(reader.ReadBytes(8));
            uint version = reader.ReadUInt32();
            uint vertexCountRaw = reader.ReadUInt32();
            uint indexCountRaw = reader.ReadUInt32();
            uint flags = reader.ReadUInt32();
            if (magic != "YXCARM01" || version != 1)
                throw new InvalidDataException("不支持的 AR Mesh 格式");
            if (
                vertexCountRaw == 0
                || indexCountRaw == 0
                || indexCountRaw % 3 != 0
                || vertexCountRaw > (uint)Mathf.Max(1, maxMobileMeshVertices)
                || indexCountRaw > (uint)Mathf.Max(3, maxMobileMeshIndices)
            )
                throw new InvalidDataException("AR Mesh 顶点或索引数量超出手机客户端限制");

            long expectedBytes = 24L + (long)vertexCountRaw * 24L + (long)indexCountRaw * 4L;
            if (payload.LongLength != expectedBytes)
                throw new InvalidDataException("AR Mesh 字节数与头部不一致");

            int vertexCount = (int)vertexCountRaw;
            int indexCount = (int)indexCountRaw;
            vertices = new Vector3[vertexCount];
            normals = new Vector3[vertexCount];
            for (int i = 0; i < vertexCount; i++)
            {
                Vector3 vertex = new Vector3(
                    reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()
                );
                if (!IsFinite(vertex))
                    throw new InvalidDataException("AR Mesh 包含无效顶点");
                vertices[i] = vertex;
            }
            bool hasNormals = (flags & 1u) != 0;
            for (int i = 0; i < vertexCount; i++)
            {
                Vector3 normal = new Vector3(
                    reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()
                );
                normals[i] = IsFinite(normal) ? normal : Vector3.up;
            }
            triangles = new int[indexCount];
            for (int i = 0; i < indexCount; i++)
            {
                uint index = reader.ReadUInt32();
                if (index >= vertexCountRaw)
                    throw new InvalidDataException("AR Mesh 索引越界");
                triangles[i] = (int)index;
            }
            if (!hasNormals) normals = null;
        }
    }

    static bool IsFinite(Vector3 value)
    {
        return
            !float.IsNaN(value.x) && !float.IsInfinity(value.x)
            && !float.IsNaN(value.y) && !float.IsInfinity(value.y)
            && !float.IsNaN(value.z) && !float.IsInfinity(value.z);
    }

    bool CreateCaptureReferenceAnchor()
    {
        ClearCaptureReferenceAnchor();
        if (
            !hasCameraFrameSnapshot
            || ARSession.state != ARSessionState.SessionTracking
            || anchorManager == null
        )
            return false;

        Vector3 requestedPosition = cameraFramePosition;
        Quaternion requestedRotation = cameraFrameRotation;
        try
        {
            captureReferenceAnchor = anchorManager.AddAnchor(
                new Pose(
                    requestedPosition,
                    requestedRotation
                )
            );
            if (captureReferenceAnchor == null) return false;
            captureReferenceAnchorObject = captureReferenceAnchor.gameObject;
            captureReferenceAnchorObject.name = "CaptureReferenceARAnchor";

            // Freeze the transform actually returned by AR Foundation. The
            // requested Pose can be converted through the trackables parent,
            // so it is not a reliable baseline for later world compensation.
            captureReferenceAnchorPosition =
                captureReferenceAnchorObject.transform.position;
            captureReferenceAnchorRotation =
                captureReferenceAnchorObject.transform.rotation;
            float initialPositionDelta = Vector3.Distance(
                requestedPosition,
                captureReferenceAnchorPosition
            );
            float initialRotationDelta = Quaternion.Angle(
                requestedRotation,
                captureReferenceAnchorRotation
            );
            Debug.Log(
                $"[ARAnchor] capture baseline frozen from returned transform; "
                + $"request_delta={initialPositionDelta:F4}m/"
                + $"{initialRotationDelta:F2}deg "
                + $"tracking={captureReferenceAnchor.trackingState}"
            );
        }
        catch (System.Exception)
        {
            if (captureReferenceAnchorObject != null)
                Destroy(captureReferenceAnchorObject);
            captureReferenceAnchorObject = null;
            captureReferenceAnchor = null;
            return false;
        }

        captureReferenceAnchorPoseValid = true;
        captureReferenceAnchorDisplayStable = false;
        captureReferenceAnchorEverTracked = false;
        captureReferenceAnchorTrackingSince = -1.0f;
        lastCaptureReferenceAnchorTrackingState = captureReferenceAnchor.trackingState;
        return true;
    }

    bool HasCaptureReferenceAnchor()
    {
        return
            captureReferenceAnchorPoseValid
            && captureReferenceAnchorObject != null
            && captureReferenceAnchor != null;
    }

    void UpdateCaptureReferenceAnchorTracking()
    {
        if (!captureReferenceAnchorPoseValid) return;

        bool anchorAvailable =
            captureReferenceAnchorObject != null
            && captureReferenceAnchor != null;
        TrackingState state = anchorAvailable
            ? captureReferenceAnchor.trackingState
            : TrackingState.None;
        bool sessionTracking = ARSession.state == ARSessionState.SessionTracking;
        TrackingState effectiveState =
            anchorAvailable && !sessionTracking
                ? TrackingState.Limited
                : state;
        bool tracking =
            anchorAvailable
            && sessionTracking
            && state == TrackingState.Tracking;
        bool wasDisplayStable = captureReferenceAnchorDisplayStable;

        if (tracking)
        {
            captureReferenceAnchorEverTracked = true;
            if (captureReferenceAnchorTrackingSince < 0.0f)
                captureReferenceAnchorTrackingSince = Time.realtimeSinceStartup;
            captureReferenceAnchorDisplayStable =
                Time.realtimeSinceStartup - captureReferenceAnchorTrackingSince
                >= Mathf.Max(0.0f, reconstructedAnchorRecoveryStableSeconds);
        }
        else
        {
            captureReferenceAnchorTrackingSince = -1.0f;
            captureReferenceAnchorDisplayStable = false;
        }

        bool stateChanged =
            effectiveState != lastCaptureReferenceAnchorTrackingState;
        if (
            stateChanged
            || wasDisplayStable != captureReferenceAnchorDisplayStable
        )
        {
            ApplyReconstructedMeshDisplayMode();
        }

        if (reconstructedMeshRoot != null)
        {
            if (!tracking && stateChanged)
            {
                string message = effectiveState == TrackingState.Limited
                    ? "AR 定位暂时受限，Mesh 已隐藏；请移回采集区域"
                    : (
                        captureReferenceAnchorEverTracked
                            ? "AR Anchor 跟踪丢失，Mesh 已隐藏；请移回采集区域"
                            : "AR Anchor 正在建立，Mesh 将在定位稳定后显示"
                    );
                UpdateUI(
                    message,
                    effectiveState == TrackingState.Limited ? Color.yellow : Color.red
                );
            }
            else if (
                !wasDisplayStable
                && captureReferenceAnchorDisplayStable
            )
            {
                UpdateUI("AR Anchor 已恢复稳定，Mesh 已重新对齐显示", Color.green);
            }
        }

        lastCaptureReferenceAnchorTrackingState = effectiveState;
    }

    void ClearCaptureReferenceAnchor()
    {
        if (captureReferenceAnchorObject != null)
            Destroy(captureReferenceAnchorObject);
        captureReferenceAnchorObject = null;
        captureReferenceAnchor = null;
        captureReferenceAnchorPosition = Vector3.zero;
        captureReferenceAnchorRotation = Quaternion.identity;
        captureReferenceAnchorPoseValid = false;
        captureReferenceAnchorDisplayStable = false;
        captureReferenceAnchorEverTracked = false;
        captureReferenceAnchorTrackingSince = -1.0f;
        lastCaptureReferenceAnchorTrackingState = TrackingState.None;
    }

    void CreateReconstructedMeshOverlay(
        Vector3[] vertices,
        Vector3[] normals,
        int[] triangles
    )
    {
        ClearReconstructedMesh();
        if (!HasCaptureReferenceAnchor())
            throw new System.InvalidOperationException("采集 Anchor 不可用");

        reconstructedMeshRoot = new GameObject("ReconstructedObjectARWorld");
        reconstructedMeshRoot.transform.SetParent(
            captureReferenceAnchorObject.transform,
            false
        );
        // Vertices are expressed in the Unity world used during capture.
        // A_current * inverse(A_capture) keeps that world aligned after relocalization.
        Quaternion captureRotationInverse = Quaternion.Inverse(
            captureReferenceAnchorRotation
        );
        reconstructedMeshRoot.transform.localPosition =
            captureRotationInverse * (-captureReferenceAnchorPosition);
        reconstructedMeshRoot.transform.localRotation = captureRotationInverse;
        reconstructedMeshRoot.transform.localScale = Vector3.one;

        reconstructedSurfaceObject = new GameObject("Surface");
        reconstructedSurfaceObject.transform.SetParent(reconstructedMeshRoot.transform, false);
        MeshFilter surfaceFilter = reconstructedSurfaceObject.AddComponent<MeshFilter>();
        MeshRenderer surfaceRenderer = reconstructedSurfaceObject.AddComponent<MeshRenderer>();
        reconstructedSurfaceMesh = new Mesh();
        reconstructedSurfaceMesh.name = "ReconstructedObjectSurface";
        reconstructedSurfaceMesh.indexFormat = vertices.Length > 65535
            ? IndexFormat.UInt32
            : IndexFormat.UInt16;
        reconstructedSurfaceMesh.vertices = vertices;
        reconstructedSurfaceMesh.triangles = triangles;
        if (normals != null && normals.Length == vertices.Length)
            reconstructedSurfaceMesh.normals = normals;
        else
            reconstructedSurfaceMesh.RecalculateNormals();
        reconstructedSurfaceMesh.RecalculateBounds();
        surfaceFilter.sharedMesh = reconstructedSurfaceMesh;
        reconstructedSurfaceMaterial = CreateARMaterial(
            reconstructedSurfaceMaterialTemplate,
            reconstructedSurfaceColor,
            true
        );
        surfaceRenderer.sharedMaterial = reconstructedSurfaceMaterial;
        surfaceRenderer.shadowCastingMode = ShadowCastingMode.Off;
        surfaceRenderer.receiveShadows = false;

        reconstructedOutlineObject = new GameObject("Outline");
        reconstructedOutlineObject.transform.SetParent(reconstructedMeshRoot.transform, false);
        MeshFilter outlineFilter = reconstructedOutlineObject.AddComponent<MeshFilter>();
        MeshRenderer outlineRenderer = reconstructedOutlineObject.AddComponent<MeshRenderer>();
        reconstructedOutlineMesh = new Mesh();
        reconstructedOutlineMesh.name = "ReconstructedObjectOutline";
        reconstructedOutlineMesh.indexFormat = vertices.Length > 65535
            ? IndexFormat.UInt32
            : IndexFormat.UInt16;
        reconstructedOutlineMesh.vertices = vertices;
        reconstructedOutlineMesh.SetIndices(
            BuildUniqueEdgeIndices(triangles), MeshTopology.Lines, 0, true
        );
        outlineFilter.sharedMesh = reconstructedOutlineMesh;
        reconstructedOutlineMaterial = CreateARMaterial(
            reconstructedOutlineMaterialTemplate,
            reconstructedOutlineColor,
            false
        );
        outlineRenderer.sharedMaterial = reconstructedOutlineMaterial;
        outlineRenderer.shadowCastingMode = ShadowCastingMode.Off;
        outlineRenderer.receiveShadows = false;

        reconstructedMeshDisplayMode = 0;
        ApplyReconstructedMeshDisplayMode();
        if (meshDisplayButton != null) meshDisplayButton.gameObject.SetActive(true);
    }

    static int[] BuildUniqueEdgeIndices(int[] triangles)
    {
        HashSet<ulong> edges = new HashSet<ulong>();
        List<int> indices = new List<int>(triangles.Length * 2);
        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            AddUniqueEdge(triangles[i], triangles[i + 1], edges, indices);
            AddUniqueEdge(triangles[i + 1], triangles[i + 2], edges, indices);
            AddUniqueEdge(triangles[i + 2], triangles[i], edges, indices);
        }
        return indices.ToArray();
    }

    static void AddUniqueEdge(
        int first,
        int second,
        HashSet<ulong> edges,
        List<int> indices
    )
    {
        uint low = (uint)Mathf.Min(first, second);
        uint high = (uint)Mathf.Max(first, second);
        ulong key = ((ulong)low << 32) | high;
        if (edges.Add(key))
        {
            indices.Add((int)low);
            indices.Add((int)high);
        }
    }

    static Material CreateARMaterial(
        Material template,
        Color color,
        bool transparent
    )
    {
        Material material;
        if (template != null)
        {
            material = new Material(template);
        }
        else
        {
            // This fallback is useful in the editor, but Android builds should
            // bind explicit Material assets so their shaders are not stripped.
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null)
            {
                string slot = transparent
                    ? "Reconstructed Surface Material Template"
                    : "Reconstructed Outline Material Template";
                throw new System.InvalidOperationException(
                    $"请在 ARPoseTracker Inspector 中绑定 {slot}"
                );
            }
            material = new Material(shader);
        }
        material.name = transparent ? "ARMeshSurface" : "ARMeshOutline";
        material.color = color;
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        if (transparent)
        {
            if (material.HasProperty("_Surface")) material.SetFloat("_Surface", 1.0f);
            if (material.HasProperty("_ZWrite")) material.SetFloat("_ZWrite", 0.0f);
            if (material.HasProperty("_SrcBlend"))
                material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            if (material.HasProperty("_DstBlend"))
                material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            material.renderQueue = (int)RenderQueue.Transparent;
        }
        return material;
    }

    public void ToggleReconstructedMeshDisplay()
    {
        if (reconstructedMeshRoot == null) return;
        reconstructedMeshDisplayMode = (reconstructedMeshDisplayMode + 1) % 3;
        ApplyReconstructedMeshDisplayMode();
    }

    void ApplyReconstructedMeshDisplayMode()
    {
        bool anchorReady = captureReferenceAnchorDisplayStable;
        bool showSurface = anchorReady && reconstructedMeshDisplayMode == 0;
        bool showOutline =
            anchorReady
            && (reconstructedMeshDisplayMode == 0 || reconstructedMeshDisplayMode == 1);
        if (reconstructedSurfaceObject != null) reconstructedSurfaceObject.SetActive(showSurface);
        if (reconstructedOutlineObject != null) reconstructedOutlineObject.SetActive(showOutline);
        string label = !anchorReady
            ? "等待 AR 定位"
            : (
                reconstructedMeshDisplayMode == 0
                    ? "Mesh+轮廓"
                    : (reconstructedMeshDisplayMode == 1 ? "仅轮廓" : "隐藏 Mesh")
            );
        if (meshDisplayModeText != null) meshDisplayModeText.text = label;
        SetButtonLabel(meshDisplayButton, label);
    }

    void ClearReconstructedMesh()
    {
        if (reconstructedMeshRoot != null) Destroy(reconstructedMeshRoot);
        if (reconstructedSurfaceMesh != null) Destroy(reconstructedSurfaceMesh);
        if (reconstructedOutlineMesh != null) Destroy(reconstructedOutlineMesh);
        if (reconstructedSurfaceMaterial != null) Destroy(reconstructedSurfaceMaterial);
        if (reconstructedOutlineMaterial != null) Destroy(reconstructedOutlineMaterial);
        reconstructedMeshRoot = null;
        reconstructedSurfaceObject = null;
        reconstructedOutlineObject = null;
        reconstructedSurfaceMesh = null;
        reconstructedOutlineMesh = null;
        reconstructedSurfaceMaterial = null;
        reconstructedOutlineMaterial = null;
        if (meshDisplayButton != null) meshDisplayButton.gameObject.SetActive(false);
        if (meshDisplayModeText != null) meshDisplayModeText.text = "";
    }

    void ResetPoseDiverseCapture()
    {
        poseDiversityTargetValid = false;
        poseDiversityTargetPosition = Vector3.zero;
        acceptedPoseDiversityDirections.Clear();
        lastPoseDiversityMinimumAngle = -1.0f;
    }

    void InitializePoseDiversityTarget(Vector3 cameraPosition, Quaternion cameraRotation)
    {
        if (!enablePoseDiverseCapture)
        {
            poseDiversityTargetValid = false;
            return;
        }
        float distance = Mathf.Max(0.05f, assumedObjectDistanceMeters);
        poseDiversityTargetPosition =
            cameraPosition + cameraRotation * Vector3.forward * distance;
        poseDiversityTargetValid = true;
    }

    bool EvaluatePoseDiverseCandidate(
        Vector3 cameraPosition,
        Quaternion cameraRotation,
        out Vector3 direction,
        out float minimumAngleDegrees
    )
    {
        if (!poseDiversityTargetValid)
            InitializePoseDiversityTarget(cameraPosition, cameraRotation);

        Vector3 offset = poseDiversityTargetValid
            ? cameraPosition - poseDiversityTargetPosition
            : -(cameraRotation * Vector3.forward);
        if (offset.sqrMagnitude <= 1.0e-8f)
            offset = -(cameraRotation * Vector3.forward);
        direction = offset.normalized;

        if (!enablePoseDiverseCapture || acceptedPoseDiversityDirections.Count == 0)
        {
            minimumAngleDegrees = 180.0f;
            lastPoseDiversityMinimumAngle = -1.0f;
            return true;
        }

        minimumAngleDegrees = 180.0f;
        foreach (Vector3 accepted in acceptedPoseDiversityDirections)
            minimumAngleDegrees = Mathf.Min(
                minimumAngleDegrees,
                Vector3.Angle(accepted, direction)
            );
        lastPoseDiversityMinimumAngle = minimumAngleDegrees;
        return minimumAngleDegrees
            >= Mathf.Max(1.0f, minimumPoseDiversityAngleDegrees);
    }

    IEnumerator ProcessAndSendData(
        XRCpuImage image,
        Vector3 pos,
        Quaternion quat,
        bool hasIntrinsics,
        XRCameraIntrinsics intrinsics,
        double cpuImageTimestampSeconds,
        long frameTimestampNs,
        double poseSampleSeconds,
        double timestampDeltaSeconds,
        Matrix4x4 displayMatrix,
        Matrix4x4 projectionMatrix,
        string poseBinding,
        Vector3 poseDiversityDirection,
        float poseDiversityMinimumAngle
    )
    {
        isSending = true;
        var conversionParams = new UnityEngine.XR.ARSubsystems.XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(image.width, image.height), // 已恢复全分辨率
            outputFormat = TextureFormat.RGBA32,
            transformation = UnityEngine.XR.ARSubsystems.XRCpuImage.Transformation.None
        };
        int cpuImageWidth = image.width;
        int cpuImageHeight = image.height;

        int size = image.GetConvertedDataSize(conversionParams);
        var buffer = new NativeArray<byte>(size, Allocator.Temp);
        image.Convert(conversionParams, buffer);
        image.Dispose();

        if (cameraTexture == null || cameraTexture.width != conversionParams.outputDimensions.x || cameraTexture.height != conversionParams.outputDimensions.y)
            cameraTexture = new Texture2D(conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, TextureFormat.RGBA32, false);

        cameraTexture.LoadRawTextureData(buffer);
        cameraTexture.Apply();
        buffer.Dispose();

        byte[] imageBytes = cameraTexture.EncodeToJPG(75);
        WWWForm form = new WWWForm();
        Vector3 rot = quat.eulerAngles;
        form.AddField("pos_x", FloatString(pos.x));
        form.AddField("pos_y", FloatString(pos.y));
        form.AddField("pos_z", FloatString(pos.z));
        form.AddField("rot_x", FloatString(rot.x));
        form.AddField("rot_y", FloatString(rot.y));
        form.AddField("rot_z", FloatString(rot.z));
        form.AddField("quat_x", FloatString(quat.x));
        form.AddField("quat_y", FloatString(quat.y));
        form.AddField("quat_z", FloatString(quat.z));
        form.AddField("quat_w", FloatString(quat.w));
        form.AddField("image_width", IntString(cameraTexture.width));
        form.AddField("image_height", IntString(cameraTexture.height));
        form.AddField("cpu_image_width", IntString(cpuImageWidth));
        form.AddField("cpu_image_height", IntString(cpuImageHeight));
        form.AddField("image_transform", CpuImageTransformName);
        form.AddField("cpu_image_timestamp_s", DoubleString(cpuImageTimestampSeconds));
        form.AddField("camera_frame_timestamp_ns", frameTimestampNs.ToString(CultureInfo.InvariantCulture));
        form.AddField("pose_sample_realtime_s", DoubleString(poseSampleSeconds));
        form.AddField("camera_frame_timestamp_delta_s", DoubleString(timestampDeltaSeconds));
        form.AddField("pose_binding", poseBinding);
        form.AddField("screen_orientation", Screen.orientation.ToString());
        form.AddField("tracking_state", ARSession.state.ToString());
        form.AddField("display_matrix", MatrixString(displayMatrix));
        form.AddField("projection_matrix", MatrixString(projectionMatrix));
        form.AddField(
            "capture_view_policy",
            enablePoseDiverseCapture
                ? "online_fixed_target_spherical_min_angle_v1"
                : "fixed_interval_unfiltered"
        );
        form.AddField("capture_target_x", FloatString(poseDiversityTargetPosition.x));
        form.AddField("capture_target_y", FloatString(poseDiversityTargetPosition.y));
        form.AddField("capture_target_z", FloatString(poseDiversityTargetPosition.z));
        form.AddField("capture_direction_x", FloatString(poseDiversityDirection.x));
        form.AddField("capture_direction_y", FloatString(poseDiversityDirection.y));
        form.AddField("capture_direction_z", FloatString(poseDiversityDirection.z));
        form.AddField(
            "capture_minimum_angle_degrees",
            FloatString(poseDiversityMinimumAngle)
        );
        form.AddField(
            "capture_angle_threshold_degrees",
            FloatString(Mathf.Max(1.0f, minimumPoseDiversityAngleDegrees))
        );
        form.AddField(
            "capture_accepted_ordinal",
            IntString(acceptedPoseDiversityDirections.Count)
        );
        form.AddField(
            "capture_recommended_candidate_count",
            IntString(Mathf.Max(8, recommendedPoseDiverseFrameCount))
        );
        if (hasIntrinsics)
        {
            form.AddField("fx", FloatString(intrinsics.focalLength.x));
            form.AddField("fy", FloatString(intrinsics.focalLength.y));
            form.AddField("cx", FloatString(intrinsics.principalPoint.x));
            form.AddField("cy", FloatString(intrinsics.principalPoint.y));
            form.AddField("intrinsic_width", IntString(intrinsics.resolution.x));
            form.AddField("intrinsic_height", IntString(intrinsics.resolution.y));
        }
        string slamPointsJson = BuildSlamPointsJson();
        if (!string.IsNullOrEmpty(slamPointsJson))
        {
            form.AddField("slam_points_schema", "arpose_tracker_frame_points_v1");
            form.AddField("slam_points_json", slamPointsJson);
        }
        form.AddBinaryData("image", imageBytes, "frame.jpg", "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(serverURL + "/upload", form))
        {
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success)
            {
                if (enablePoseDiverseCapture)
                    acceptedPoseDiversityDirections.Add(poseDiversityDirection);
                lastPoseDiversityMinimumAngle = poseDiversityMinimumAngle;
            }
            else
            {
                UpdateUI("上传采样帧失败: " + ExtractServerMessage(www), Color.red);
            }
        }
        isSending = false;
    }

    string BuildSlamPointsJson()
    {
        if (!uploadSlamPoints || pointCloudManager == null || maxSlamPointsPerFrame <= 0)
            return "";

        SlamPointPayload payload = new SlamPointPayload();
        int stride = Mathf.Max(1, slamPointStride);
        int seen = 0;

        foreach (ARPointCloud cloud in pointCloudManager.trackables)
        {
            if (!cloud.positions.HasValue)
                continue;

            var positions = cloud.positions.Value;
            for (int i = 0; i < positions.Length; i++)
            {
                if ((seen % stride) == 0)
                {
                    Vector3 p = cloud.transform.TransformPoint(positions[i]);
                    payload.points.Add(new SlamPointData
                    {
                        x = p.x,
                        y = p.y,
                        z = p.z,
                        confidence = 1.0f
                    });
                    if (payload.points.Count >= maxSlamPointsPerFrame)
                        break;
                }
                seen++;
            }

            if (payload.points.Count >= maxSlamPointsPerFrame)
                break;
        }

        payload.point_count = payload.points.Count;
        return payload.point_count > 0 ? JsonUtility.ToJson(payload) : "";
    }

    IEnumerator SendCommand(string endpoint)
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + endpoint, ""))
            yield return www.SendWebRequest();
    }
    
    void UpdateUI(string text, Color color)
    {
        if (debugText == null) return;
        debugText.text = FormatPhoneStatusText(text);
        debugText.color = color;
        debugText.rectTransform.SetAsLastSibling();
    }

    void ConfigureStatusTextForPortrait()
    {
        if (debugText == null) return;
        debugText.alignment = TextAnchor.UpperLeft;
        debugText.horizontalOverflow = HorizontalWrapMode.Wrap;
        debugText.verticalOverflow = VerticalWrapMode.Truncate;
        debugText.resizeTextForBestFit = true;
        debugText.resizeTextMinSize = Mathf.Max(8, statusTextMinFontSize);
        debugText.resizeTextMaxSize = Mathf.Max(
            debugText.resizeTextMinSize,
            statusTextMaxFontSize
        );
        debugText.raycastTarget = false;
        debugText.lineSpacing = 0.95f;
        debugText.rectTransform.SetAsLastSibling();

        lastStatusSafeArea = Screen.safeArea;
        if (!fitStatusTextToPortraitSafeArea || Screen.width <= 0 || Screen.height <= 0)
            return;
        Rect safe = Screen.safeArea;
        float safeLeft = safe.xMin / Screen.width;
        float safeRight = safe.xMax / Screen.width;
        float safeBottom = safe.yMin / Screen.height;
        float safeTop = safe.yMax / Screen.height;
        float top = Mathf.Clamp(safeTop - 0.02f, 0.0f, 1.0f);
        float bottom = Mathf.Max(
            safeBottom + 0.02f,
            top - Mathf.Clamp(statusTextHeightRatio, 0.15f, 0.60f)
        );
        RectTransform rect = debugText.rectTransform;
        rect.anchorMin = new Vector2(
            Mathf.Clamp01(safeLeft + 0.025f),
            Mathf.Clamp01(bottom)
        );
        rect.anchorMax = new Vector2(
            Mathf.Clamp01(safeRight - 0.025f),
            Mathf.Clamp01(top)
        );
        rect.pivot = new Vector2(0.5f, 1.0f);
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;
    }

    string FormatPhoneStatusText(string text)
    {
        string value = string.IsNullOrEmpty(text)
            ? ""
            : text.Replace("\r", "").Trim();
        int limit = Mathf.Max(80, statusTextMaxCharacters);
        if (value.Length <= limit) return value;
        const string suffix = "\n…详细信息已记录在服务端日志";
        int keep = Mathf.Max(1, limit - suffix.Length);
        return value.Substring(0, keep).TrimEnd() + suffix;
    }

    static string FloatString(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    static string IntString(int value)
    {
        return value.ToString(CultureInfo.InvariantCulture);
    }

    static string DoubleString(double value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    static string MatrixString(Matrix4x4 matrix)
    {
        string[] values = new string[16];
        int index = 0;
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                values[index++] = FloatString(matrix[row, column]);
            }
        }
        return string.Join(" ", values);
    }

    static string ExtractServerMessage(UnityWebRequest request)
    {
        if (request == null) return "";
        string body = request.downloadHandler != null ? request.downloadHandler.text : "";
        if (!string.IsNullOrEmpty(body))
        {
            try
            {
                ServerStatusResponse response = JsonUtility.FromJson<ServerStatusResponse>(body);
                if (response != null && !string.IsNullOrEmpty(response.message))
                    return response.message;
            }
            catch
            {
                // Fall back to Unity's transport error below.
            }
        }
        return string.IsNullOrEmpty(request.error) ? body : request.error;
    }
}
