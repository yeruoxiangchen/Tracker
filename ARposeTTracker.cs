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
using System.Security.Cryptography;
using System.Text;

[System.Serializable]
public class SelectionData
{
    public List<int> selected;
    public string session_id;
    public int lifecycle_generation;
    public string runtime_o_sha256;
    public string requested_pose_sha256;
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
    public string mesh_sha256;
    public int byte_count;
    public int vertex_count;
    public int triangle_count;
    public string coordinate_frame;
    public string placement;
    public string session_id;
    public int lifecycle_generation;
    public string runtime_o_sha256;
    public string requested_pose_sha256;
}

[System.Serializable]
public class RuntimeOPrepareResponse
{
    public string status;
    public string message;
    public string session_id;
    public int lifecycle_generation;
    public string runtime_o_sha256;
    public string requested_pose_sha256;
    public bool model_inference_started;
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
public class MeshTransformUnity
{
    public float position_x;
    public float position_y;
    public float position_z;
    public float quaternion_x;
    public float quaternion_y;
    public float quaternion_z;
    public float quaternion_w = 1.0f;
    public float uniform_scale = 1.0f;
}

[System.Serializable]
public class AlignmentRefineRequest
{
    public string session_id;
    public int lifecycle_generation;
    public string runtime_o_sha256;
    public string requested_pose_sha256;
    public string refinement_id;
    public MeshTransformUnity current_mesh_transform_unity;
}

[System.Serializable]
public class AlignmentRefineStartResponse
{
    public string status;
    public string message;
    public string session_id;
    public int lifecycle_generation;
    public string refinement_id;
    public int minimum_frames;
    public int recommended_frames;
    public int optimization_views;
}

[System.Serializable]
public class AlignmentRefineOptimizeResponse
{
    public string status;
    public string message;
    public string session_id;
    public int lifecycle_generation;
    public string refinement_id;
    public bool accepted;
    public bool geometry_regenerated;
    public MeshTransformUnity selected_mesh_transform_unity;
    public float initial_iou_mean;
    public float optimized_iou_mean;
    public float iou_gain_mean;
    public string report;
}

[System.Serializable]
public class MobileOverlayAuditRequest
{
    public string session_id;
    public int lifecycle_generation;
    public string runtime_o_sha256;
    public string requested_pose_sha256;
    public int maximum_frames;
    public bool strict_reconstruction_input_pose_matching;
    public float target_translation_tolerance_meters;
    public float target_rotation_tolerance_degrees;
    public string diagnostic_stage;
    public string alignment_refinement_state;
    public bool last_alignment_refinement_accepted;
    public string last_alignment_refinement_report;
    public MeshTransformUnity current_mesh_transform_unity;
}

[System.Serializable]
public class MobileOverlayPoseTarget
{
    public int target_index;
    public string source_frame_name;
    public string source_image_sha256;
    public float position_x;
    public float position_y;
    public float position_z;
    public float quaternion_x;
    public float quaternion_y;
    public float quaternion_z;
    public float quaternion_w = 1.0f;
}

[System.Serializable]
public class MobileOverlayAuditStartResponse
{
    public string status;
    public string message;
    public string session_id;
    public int lifecycle_generation;
    public string audit_id;
    public int maximum_frames;
    public bool strict_reconstruction_input_pose_matching;
    public float target_translation_tolerance_meters;
    public float target_rotation_tolerance_degrees;
    public MobileOverlayPoseTarget[] pose_targets;
    public bool diagnostic_only;
    public string report;
}

[System.Serializable]
public class MobileOverlayAuditUploadResponse
{
    public string status;
    public string session_id;
    public string audit_id;
    public int captured_frames;
    public int maximum_frames;
    public bool complete;
    public bool diagnostic_only;
    public int matched_target_index;
    public string matched_target_source_frame_name;
    public string report;
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
    private const string MobileOverlayAuditContract =
        "unity_native_screen_display_aligned_raw_rgb_strict_input_pose_v3";

    private enum AlignmentRefinementState
    {
        Unavailable = 0,
        Ready = 1,
        Capturing = 2,
        Optimizing = 3,
        Complete = 4,
    }
    private enum ReconstructedOutlineMethod
    {
        ViewDependentMeshLines = 0,
        ServerStyleScreenSpace = 1,
    }

    private sealed class SilhouetteEdge
    {
        public int firstVertex;
        public int secondVertex;
        public int firstFace;
        public int secondFace = -1;
    }

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
    [Tooltip("第一阶段客户端固定每0.2秒尝试上传一帧；Start中再次强制，避免旧场景序列化值覆盖")]
    public float sendInterval = 0.2f;
    [Tooltip("仅保留旧场景兼容；客户端无筛帧模式只记录时间差，不按该值丢帧")]
    public float maxCameraFrameTimestampDeltaSeconds = 0.0f;

    [Header("AR 轨迹稳定性")]
    public float trackingWarmupSeconds = 1.5f;
    public float trackingResetTimeoutSeconds = 15.0f;
    public float maxConsecutiveCameraPoseJumpMeters = 0.75f;
    public float maxConsecutiveCameraPoseJumpDegrees = 45.0f;

    [Header("固定间隔无筛选采集")]
    [Tooltip("旧场景兼容字段；运行时强制为false，任何姿态角都不会导致丢帧")]
    public bool enablePoseDiverseCapture = false;
    [Tooltip("录制开始时沿首帧相机前向估计物体中心的距离（米）")]
    public float assumedObjectDistanceMeters = 0.70f;
    [Range(1.0f, 45.0f)]
    public float minimumPoseDiversityAngleDegrees = 10.0f;
    [Tooltip("UI建议的候选帧数；最终模型输入仍由服务端筛成8视角")]
    public int recommendedPoseDiverseFrameCount = 24;

    [Header("重建 Mesh AR 显示")]
    public Button meshDisplayButton;
    public Text meshDisplayModeText;
    [Tooltip("可选：绑定自定义按钮；留空时会在运行时复制 Mesh Display Button 生成切换按钮")]
    public Button meshOutlineMethodButton;
    public Text meshOutlineMethodText;
    [Tooltip("请在 Unity Inspector 中绑定使用 URP/Unlit 的轮廓材质，避免 Android 构建裁剪 Shader")]
    public Material reconstructedOutlineMaterialTemplate;
    [Tooltip("服务器式轮廓的白色 silhouette mask 材质（Tracker/ARMeshSilhouetteMask）")]
    public Material serverStyleMaskMaterialTemplate;
    [Tooltip("服务器式屏幕空间外轮廓材质（Tracker/ARMeshScreenSpaceOutline）")]
    public Material serverStyleOutlineMaterialTemplate;
    public Color reconstructedOutlineColor = new Color(0.05f, 1.0f, 0.70f, 1.0f);
    [Tooltip("根据当前相机位置重算视角相关剪影边的时间间隔（秒）")]
    public float reconstructedSilhouetteUpdateIntervalSeconds = 0.10f;
    [Tooltip("真实 ARAnchor 连续处于 Tracking 多久后才恢复显示 Mesh；Limited 时始终隐藏")]
    public float reconstructedAnchorRecoveryStableSeconds = 0.75f;
    [Range(0.25f, 1.0f)]
    [Tooltip("服务器式轮廓的离屏渲染分辨率；0.5 的像素数是全屏的 1/4")]
    public float serverStyleRenderScale = 0.50f;
    [Range(1.0f, 6.0f)]
    public float serverStyleOutlineWidthPixels = 3.0f;
    [Range(5.0f, 60.0f)]
    [Tooltip("服务器式 mask 最大刷新率；屏幕仍每帧显示上一张轮廓，默认30fps以控制发热")]
    public float serverStyleMaxFramesPerSecond = 30.0f;
    [Range(0, 31)]
    [Tooltip("专用于 silhouette mask 的 Unity Layer，请确保不被其他场景对象使用")]
    public int serverStyleOutlineLayer = 31;
    public int maxMobileMeshVertices = 200000;
    public int maxMobileMeshIndices = 600000;

    [Header("统一底部按钮布局")]
    [Tooltip("录制/重新录制固定为最底部最大主按钮；Mesh 三项操作固定在它上方")]
    public bool useBottomSafeAreaMeshControlDock = true;
    [Range(64.0f, 120.0f)] public float meshControlButtonHeight = 74.0f;
    [Range(4.0f, 28.0f)] public float meshControlButtonSpacing = 12.0f;
    [Range(0.0f, 0.10f)] public float meshControlDockSideMarginRatio = 0.035f;
    [Range(0.0f, 0.06f)] public float meshControlDockBottomMarginRatio = 0.012f;
    public int meshControlButtonMinFontSize = 17;
    public int meshControlButtonMaxFontSize = 28;
    public Color meshControlDockColor = new Color(0.02f, 0.04f, 0.07f, 0.72f);
    [Range(88.0f, 170.0f)] public float primaryRecordButtonHeight = 118.0f;
    [Range(0.0f, 0.08f)] public float primaryRecordSideMarginRatio = 0.035f;
    [Range(0.0f, 0.08f)] public float primaryRecordBottomMarginRatio = 0.018f;
    [Range(4.0f, 32.0f)] public float bottomDockVerticalSpacing = 12.0f;
    public int primaryRecordMinFontSize = 22;
    public int primaryRecordMaxFontSize = 36;
    public Color primaryRecordColor = new Color(0.02f, 0.52f, 0.40f, 0.98f);
    public Color secondaryActionColor = new Color(0.12f, 0.28f, 0.48f, 0.96f);
    public Color destructiveActionColor = new Color(0.72f, 0.16f, 0.18f, 0.96f);

    [Header("返回现场快速 O2W/A0 Mesh 校准")]
    [Tooltip("可选：留空时自动复制 Mesh 显示按钮；第一次按下采集，第二次按下优化")]
    public Button alignmentRefineButton;
    [Tooltip("快速校准帧间隔；与首次录制隔离")]
    public float alignmentRefineSendInterval = 0.25f;
    [Tooltip("候选帧没有上限；少于16帧不会启动优化")]
    public int alignmentRefineMinimumFrames = 16;
    [Tooltip("建议至少采集32个候选；服务端从全部候选中球面FPS选16帧")]
    public int alignmentRefineRecommendedFrames = 32;

    [Header("显式手机位姿诊断录制（仅诊断）")]
    [Tooltip("可选：留空时自动复制 Mesh 显示按钮；Mesh 返回后可在快速校准前后分别录制")]
    public Button poseDiagnosticRecordButton;
    [Tooltip("总开关；关闭时诊断按钮不可用")]
    public bool enableMobileOverlayAudit = true;
    [Tooltip("默认关闭。启用后保持旧行为：Mesh显示及校准完成后自动开始诊断；通常应使用显式按钮")]
    public bool autoStartMobileOverlayAudit = false;
    [Tooltip("命中一个历史重建位姿后，到下一次接受采集的最短间隔（秒）")]
    [Range(0.5f, 5.0f)] public float mobileOverlayAuditIntervalSeconds = 1.0f;
    [Range(8, 8)] public int mobileOverlayAuditMaxFrames = 8;
    [Tooltip("现场相机与重建输入相机中心的最大误差（米）")]
    [Range(0.005f, 0.10f)] public float mobileOverlayTargetTranslationToleranceMeters = 0.025f;
    [Tooltip("现场相机与重建输入相机旋转的最大误差（度）")]
    [Range(0.5f, 15.0f)] public float mobileOverlayTargetRotationToleranceDegrees = 3.0f;
    [Tooltip("必须保持开启：手机最终画面以原生屏幕分辨率、无损 PNG 回传")]
    public bool mobileOverlayAuditKeepNativeScreenResolution = true;
    [Tooltip("仅在关闭原生分辨率时使用的长边上限")]
    [Range(1080, 4096)] public int mobileOverlayAuditMaxLongEdge = 4096;
    [Tooltip("原始 XRCpuImage 的 JPEG 质量；最终手机屏幕画面不使用 JPEG")]
    [Range(80, 100)] public int mobileOverlayAuditJpegQuality = 95;

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
    private GameObject reconstructedOutlineObject;
    private Mesh reconstructedOutlineMesh;
    private Material reconstructedOutlineMaterial;
    private int reconstructedMeshDisplayMode = 0;
    private ReconstructedOutlineMethod reconstructedOutlineMethod =
        ReconstructedOutlineMethod.ViewDependentMeshLines;
    private GameObject reconstructedServerMaskObject;
    private Mesh reconstructedServerMaskMesh;
    private Material reconstructedServerMaskMaterial;
    private GameObject reconstructedServerMaskCameraObject;
    private Camera reconstructedServerMaskCamera;
    private RenderTexture reconstructedServerMaskTexture;
    private GameObject reconstructedServerOutlineCanvasObject;
    private RawImage reconstructedServerOutlineImage;
    private Material reconstructedServerOutlineMaterial;
    private Camera reconstructedARRenderCamera;
    private int reconstructedServerMaskTextureWidth = 0;
    private int reconstructedServerMaskTextureHeight = 0;
    private int reconstructedMainCameraOriginalCullingMask = 0;
    private bool reconstructedMainCameraCullingMaskCaptured = false;
    private bool reconstructedServerStyleAvailable = false;
    private float nextServerStyleRenderSeconds = -1.0f;
    private Vector3[] reconstructedOutlineVertices;
    private int[] reconstructedOutlineTriangles;
    private List<SilhouetteEdge> reconstructedSilhouetteEdges;
    private bool[] reconstructedTriangleFrontFacing;
    private readonly List<int> reconstructedSilhouetteLineIndices = new List<int>();
    private float nextReconstructedSilhouetteUpdateSeconds = -1.0f;
    private GameObject captureReferenceAnchorObject;
    private ARAnchor captureReferenceAnchor;
    private bool captureReferenceUsesTrackedARAnchor = false;
    private Vector3 captureReferenceAnchorPosition;
    private Quaternion captureReferenceAnchorRotation = Quaternion.identity;
    private bool captureReferenceAnchorPoseValid = false;
    private bool captureReferenceAnchorTrackingStable = false;
    private bool captureReferenceAnchorEverTracked = false;
    private float captureReferenceAnchorTrackingSince = -1.0f;
    private TrackingState lastCaptureReferenceAnchorTrackingState = TrackingState.None;
    private string activeServerSessionId = "";
    private string preparedRuntimeOSha256 = "";
    private string preparedRequestedPoseSha256 = "";
    private int preparedLifecycleGeneration = -1;
    private Vector3[] pendingReconstructedVertices;
    private Vector3[] pendingReconstructedNormals;
    private int[] pendingReconstructedTriangles;
    private MobileARResponse pendingMobileARResponse;
    private MobileARResponse activeMobileARResponse;
    private AlignmentRefinementState alignmentRefinementState =
        AlignmentRefinementState.Unavailable;
    private string activeAlignmentRefinementId = "";
    private int alignmentRefinementUploadedCount = 0;
    private float alignmentRefinementTimer = 0.0f;
    private bool lastAlignmentRefinementAccepted = false;
    private string lastAlignmentRefinementReport = "";
    private bool applicationPaused = false;
    private bool mobileOverlayAuditStartPending = false;
    private bool mobileOverlayAuditStartInFlight = false;
    private bool mobileOverlayAuditCaptureActive = false;
    private bool mobileOverlayAuditSending = false;
    private string activeMobileOverlayAuditId = "";
    private int mobileOverlayAuditUploadedCount = 0;
    private int mobileOverlayAuditServerMaximumFrames = 0;
    private float mobileOverlayAuditNextCaptureSeconds = -1.0f;
    private MobileOverlayPoseTarget[] mobileOverlayAuditPoseTargets = null;
    private bool[] mobileOverlayAuditTargetCaptured = null;
    private float mobileOverlayAuditServerTranslationToleranceMeters = 0.025f;
    private float mobileOverlayAuditServerRotationToleranceDegrees = 3.0f;
    private float mobileOverlayAuditNextGuidanceUpdateSeconds = -1.0f;
    private int mobileOverlayAuditGeneration = 0;
    private string mobileOverlayAuditDiagnosticStage = "";
    // Monotonically identifies the currently owned AR capture lifecycle.  A
    // new recording (or an explicit cancel) invalidates every preparation
    // coroutine from the preceding run so an old teardown/Anchor callback cannot
    // mutate the next run.
    private int arSessionLifecycleGeneration = 0;
    private bool arSessionLifecycleTransitionInProgress = false;
    // A fresh camera frame after the preceding Mesh/Anchor teardown is the
    // observable boundary between capture generations.  Normal re-recording
    // deliberately keeps the same ARSession so ARCore can retain and improve
    // the environment map used for later relocalization.
    private int cameraFrameSequence = 0;
    private Rect lastStatusSafeArea = new Rect(-1.0f, -1.0f, -1.0f, -1.0f);
    private GameObject primaryActionDockObject;
    private RectTransform primaryActionDockRect;
    private GameObject meshControlDockObject;
    private RectTransform meshControlDockRect;
    private HorizontalLayoutGroup meshControlDockLayout;
    private Rect lastMeshControlSafeArea = new Rect(-1.0f, -1.0f, -1.0f, -1.0f);
    private int lastMeshControlScreenWidth = -1;
    private int lastMeshControlScreenHeight = -1;

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
    private bool cameraFrameHasCaptureAnchorPose = false;
    private Vector3 cameraFrameCaptureAnchorPosition;
    private Quaternion cameraFrameCaptureAnchorRotation = Quaternion.identity;
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
        // These assignments intentionally override values serialized by an old
        // Unity scene/prefab.  The phase-one capture contract is fixed-rate and
        // filter-free; Inspector state must not silently restore old gates.
        sendInterval = 0.2f;
        enablePoseDiverseCapture = false;
        maxCameraFrameTimestampDeltaSeconds = 0.0f;
        // Old serialized Inspector values downscaled diagnostics to 1080 px
        // and JPEG quality 85.  The diagnostic protocol now requires the
        // native end-of-frame screen as lossless PNG.
        mobileOverlayAuditMaxFrames = 8;
        mobileOverlayAuditKeepNativeScreenResolution = true;
        mobileOverlayAuditMaxLongEdge = Mathf.Max(
            4096,
            mobileOverlayAuditMaxLongEdge
        );
        mobileOverlayAuditJpegQuality = Mathf.Max(
            95,
            mobileOverlayAuditJpegQuality
        );
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
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
        ConfigureUnifiedReviewButtons();
        EnsurePrimaryActionDock();
        EnsureMeshOutlineMethodButton();
        EnsureAlignmentRefineButton();
        EnsurePoseDiagnosticRecordButton();
        EnsureMeshControlDock();
        ConfigureBottomActionDocksForSafeArea();
        if (meshDisplayButton != null)
        {
            meshDisplayButton.onClick.AddListener(ToggleReconstructedMeshDisplay);
            meshDisplayButton.gameObject.SetActive(false);
        }
        if (meshOutlineMethodButton != null)
        {
            meshOutlineMethodButton.onClick.AddListener(ToggleReconstructedOutlineMethod);
            meshOutlineMethodButton.gameObject.SetActive(false);
        }
        if (alignmentRefineButton != null)
        {
            alignmentRefineButton.onClick.AddListener(
                ToggleFastAlignmentRefinement
            );
            alignmentRefineButton.gameObject.SetActive(false);
        }
        if (poseDiagnosticRecordButton != null)
        {
            poseDiagnosticRecordButton.onClick.AddListener(
                StartExplicitPoseDiagnosticRecording
            );
            poseDiagnosticRecordButton.gameObject.SetActive(false);
        }
        RefreshPrimaryActionDockVisibility();
        RefreshMeshControlDockVisibility();

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
        arSessionLifecycleGeneration++;
        Screen.sleepTimeout = SleepTimeout.SystemSetting;
        if (cameraManager != null) cameraManager.frameReceived -= OnCameraFrameReceived;
        ClearReconstructedMesh();
        ClearCaptureReferenceAnchor();
    }

    void OnApplicationPause(bool paused)
    {
        applicationPaused = paused;
        if (!captureReferenceAnchorPoseValid) return;

        // A local ARAnchor may survive a temporary screen-off/background pause,
        // but its pose must never be consumed until ARFoundation reports the
        // actual anchor as Tracking again.  Keep the anchor object and only
        // reset the continuous-Tracking recovery timer.
        captureReferenceAnchorTrackingSince = -1.0f;
        captureReferenceAnchorTrackingStable = false;
        ApplyReconstructedMeshDisplayMode();
        if (reconstructedMeshRoot != null)
        {
            UpdateUI(
                paused
                    ? "应用已暂停，Mesh 已隐藏；返回后需重新定位 Anchor"
                    : "已返回 AR，正在等待 Anchor 恢复 Tracking...",
                Color.yellow
            );
        }
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        if (arCamera == null) return;
        cameraFrameSequence++;
        Vector3 currentPosition = arCamera.position;
        Quaternion currentRotation = arCamera.rotation;
        bool hasA0Pose = TryGetPoseRelativeToCaptureAnchor(
            currentPosition,
            currentRotation,
            out Vector3 currentA0Position,
            out Quaternion currentA0Rotation
        );
        if (isRecording && captureWorldFrameValid && hasPreviousCaptureWorldPose && hasA0Pose)
        {
            float poseStep = Vector3.Distance(
                previousCaptureWorldCameraPosition, currentA0Position
            );
            float rotationStep = Quaternion.Angle(
                previousCaptureWorldCameraRotation, currentA0Rotation
            );
            if (
                (maxConsecutiveCameraPoseJumpMeters > 0.0f
                    && poseStep > maxConsecutiveCameraPoseJumpMeters)
                || (maxConsecutiveCameraPoseJumpDegrees > 0.0f
                    && rotationStep > maxConsecutiveCameraPoseJumpDegrees)
            )
            {
                // Diagnostic only.  Do not cancel the capture or drop this frame;
                // server2 must receive the unfiltered chronological trajectory.
                UpdateUI(
                    $"记录到 AR 位姿跳变 {poseStep:F2} m / {rotationStep:F1}°；"
                    + "无筛帧模式继续采集",
                    Color.yellow
                );
            }
        }
        if (isRecording && captureWorldFrameValid && hasA0Pose)
        {
            previousCaptureWorldCameraPosition = currentA0Position;
            previousCaptureWorldCameraRotation = currentA0Rotation;
            hasPreviousCaptureWorldPose = true;
        }
        cameraFramePosition = currentPosition;
        cameraFrameRotation = currentRotation;
        cameraFrameHasCaptureAnchorPose = hasA0Pose;
        if (hasA0Pose)
        {
            cameraFrameCaptureAnchorPosition = currentA0Position;
            cameraFrameCaptureAnchorRotation = currentA0Rotation;
        }
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
        if (isPreparingRecording || arSessionLifecycleTransitionInProgress) return;
        if (!isRecording)
        {
            StartCoroutine(BeginRecordingWithFreshCaptureGeneration());
        }
        else
        {
            isRecording = false;
            recordButton.gameObject.SetActive(false); 
            UpdateUI(
                HasCaptureReferenceFrame()
                    ? "采集结束；A0 参考 Anchor 保持不变，正在准备原图点选..."
                    : "采集结束，但录制起点 A0 已丢失；拒绝继续生成错位 Mesh",
                HasCaptureReferenceFrame() ? Color.yellow : Color.red
            );
            if (HasCaptureReferenceFrame())
                StartCoroutine(RequestPreprocess());
            else
            {
                recordButton.gameObject.SetActive(true);
                buttonText.text = "重新录制";
                buttonText.color = Color.white;
            }
        }
    }

    IEnumerator BeginRecordingWithFreshCaptureGeneration()
    {
        int lifecycleGeneration = ++arSessionLifecycleGeneration;
        arSessionLifecycleTransitionInProgress = true;
        isPreparingRecording = true;
        isRecording = false;
        captureWorldFrameValid = false;
        hasPreviousCaptureWorldPose = false;
        ResetPoseDiverseCapture();

        // Capture the provider TrackableIds before clearing the managed
        // references. Destroy is deferred until the end of the frame. The
        // next run must not create A0 until the preceding A0 has disappeared
        // from ARAnchorManager.
        bool hadPreviousA0 = captureReferenceAnchor != null;
        TrackableId previousA0Id = hadPreviousA0
            ? captureReferenceAnchor.trackableId
            : default(TrackableId);

        ClearReconstructedMesh();
        ClearCaptureReferenceAnchor();
        ClearPendingReconstructedMesh();
        activeServerSessionId = "";
        preparedRuntimeOSha256 = "";
        preparedRequestedPoseSha256 = "";
        preparedLifecycleGeneration = -1;
        hasCameraFrameSnapshot = false;
        cameraFrameHasCaptureAnchorPose = false;
        timer = 0f;
        segmentationReady = false;
        addForegroundPoint = true;
        approvedSeedFrames.Clear();
        recordButton.interactable = false;
        buttonText.text = "正在清理旧 Anchor 并稳定 AR...";
        buttonText.color = Color.yellow;

        // Let Unity finish deferred destruction. Do not call ARSession.Reset
        // for a normal new recording: Reset destroys every trackable and the
        // device-tracking map, which makes the second and later reconstruction
        // less able to relocalize after a long inference wait.
        yield return null;
        yield return new WaitForEndOfFrame();
        if (lifecycleGeneration != arSessionLifecycleGeneration)
            yield break;

        if (arSession == null)
        {
            arSessionLifecycleTransitionInProgress = false;
            isPreparingRecording = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI("AR Session 未绑定，无法建立新的采集世代", Color.red);
            yield break;
        }

        int preTransitionCameraFrameSequence = cameraFrameSequence;
        float transitionRequestedSeconds = Time.realtimeSinceStartup;
        float deadline = Time.realtimeSinceStartup + Mathf.Max(1.0f, trackingResetTimeoutSeconds);
        float stableSince = -1.0f;
        bool previousA0Removed = !hadPreviousA0;
        bool freshCameraFrameObserved = false;
        while (Time.realtimeSinceStartup < deadline)
        {
            if (lifecycleGeneration != arSessionLifecycleGeneration)
                yield break;

            previousA0Removed =
                !hadPreviousA0 || !AnchorTrackableStillRegistered(previousA0Id);
            freshCameraFrameObserved =
                hasCameraFrameSnapshot
                && cameraFrameSequence > preTransitionCameraFrameSequence
                && cameraFramePoseSampleSeconds >= transitionRequestedSeconds;
            bool stable =
                ARSession.state == ARSessionState.SessionTracking
                && freshCameraFrameObserved
                && previousA0Removed;
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
            arSessionLifecycleTransitionInProgress = false;
            isPreparingRecording = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI(
                "新采集世代隔离失败："
                + $"旧A0移除={previousA0Removed}, "
                + $"新相机帧={freshCameraFrameObserved}, "
                + $"状态={ARSession.state}；请重试，禁止复用旧会话",
                Color.red
            );
            yield break;
        }

        Debug.Log(
            $"[ARCaptureLifecycle] generation={lifecycleGeneration} isolated; "
            + $"old_a0_removed={previousA0Removed} "
            + "session_reset=false "
            + $"fresh_camera_sequence={cameraFrameSequence}"
        );

        bool anchorCreated = CreateCaptureReferenceAnchor();
        if (!anchorCreated)
        {
            arSessionLifecycleTransitionInProgress = false;
            isPreparingRecording = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI("无法创建录制起点 A0 Anchor，请保持相机稳定后重试", Color.red);
            yield break;
        }

        float anchorDeadline = Time.realtimeSinceStartup
            + Mathf.Max(1.0f, trackingResetTimeoutSeconds);
        float anchorStableSince = -1.0f;
        while (Time.realtimeSinceStartup < anchorDeadline)
        {
            if (lifecycleGeneration != arSessionLifecycleGeneration)
                yield break;
            bool a0Tracking = CaptureReferencePoseUsable();
            if (a0Tracking)
            {
                if (anchorStableSince < 0.0f)
                    anchorStableSince = Time.realtimeSinceStartup;
                if (
                    cameraFrameHasCaptureAnchorPose
                    && Time.realtimeSinceStartup - anchorStableSince
                        >= Mathf.Max(0.0f, reconstructedAnchorRecoveryStableSeconds)
                )
                    break;
            }
            else
            {
                anchorStableSince = -1.0f;
            }
            yield return null;
        }
        if (
            !CaptureReferencePoseUsable()
            || !cameraFrameHasCaptureAnchorPose
            || anchorStableSince < 0.0f
            || Time.realtimeSinceStartup - anchorStableSince
                < Mathf.Max(0.0f, reconstructedAnchorRecoveryStableSeconds)
        )
        {
            ClearCaptureReferenceAnchor();
            arSessionLifecycleTransitionInProgress = false;
            isPreparingRecording = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI("A0 Anchor 未能稳定 Tracking，请对准有纹理环境后重试", Color.red);
            yield break;
        }

        yield return StartCoroutine(SendCommand("/start_record"));
        if (lifecycleGeneration != arSessionLifecycleGeneration)
            yield break;
        InitializePoseDiversityTarget(
            cameraFrameCaptureAnchorPosition,
            cameraFrameCaptureAnchorRotation
        );
        captureWorldFrameValid = true;
        previousCaptureWorldCameraPosition = cameraFrameCaptureAnchorPosition;
        previousCaptureWorldCameraRotation = cameraFrameCaptureAnchorRotation;
        hasPreviousCaptureWorldPose = true;
        isRecording = true;
        isPreparingRecording = false;
        arSessionLifecycleTransitionInProgress = false;
        recordButton.interactable = true;
        buttonText.text = "结束录制并分割";
        buttonText.color = Color.red;
        UpdateUI(
            $"连续 AR Session 的采集世代 G{lifecycleGeneration} 已隔离，A0 Anchor 已稳定；"
            + "开始上传 A0-relative 相机位姿并全程采集",
            Color.green
        );
    }

    bool AnchorTrackableStillRegistered(TrackableId trackableId)
    {
        if (anchorManager == null) return false;
        try
        {
            return anchorManager.GetAnchor(trackableId) != null;
        }
        catch (System.Exception exception)
        {
            // Treat an unreadable manager state as not-yet-isolated.  The
            // preparation timeout will stop the run rather than silently
            // mixing two AR session generations.
            Debug.LogWarning(
                "[ARSessionLifecycle] failed to query old Anchor: "
                + exception.Message
            );
            return true;
        }
    }

    // ========== 更新：全局取消与重置 ==========
    void CancelReview()
    {
        // Explicit exit is the only UI path that performs a hard ARSession
        // Reset. Normal re-recording deliberately keeps the environment map.
        int lifecycleGeneration = ++arSessionLifecycleGeneration;
        arSessionLifecycleTransitionInProgress = true;
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
        ClearPendingReconstructedMesh();
        activeServerSessionId = "";
        preparedRuntimeOSha256 = "";
        preparedRequestedPoseSha256 = "";
        preparedLifecycleGeneration = -1;
        
        // 2. 隐藏审核面板（如果它正开着的话）
        if (reviewPanel != null) reviewPanel.SetActive(false);
        
        // 3. 在 Reset 完成前锁住主界面的录制按钮
        recordButton.gameObject.SetActive(true);
        recordButton.interactable = false;
        buttonText.text = "正在重置 AR Session...";
        buttonText.color = Color.yellow;
        
        UpdateUI("正在退出本轮并硬重置 AR Session...", Color.yellow);

        // 4. 通知服务器中断当前任务并清空缓存
        StartCoroutine(SendCommand("/cancel_review"));
        StartCoroutine(ResetARSessionAfterExplicitCancel(lifecycleGeneration));
    }

    IEnumerator ResetARSessionAfterExplicitCancel(int lifecycleGeneration)
    {
        // Let Unity finish deferred Mesh/A0 destruction before ARFoundation
        // destroys provider trackables and starts a fresh session generation.
        yield return null;
        yield return new WaitForEndOfFrame();
        if (lifecycleGeneration != arSessionLifecycleGeneration)
            yield break;

        if (arSession == null)
        {
            arSessionLifecycleTransitionInProgress = false;
            recordButton.interactable = true;
            buttonText.text = "开始录制";
            buttonText.color = Color.white;
            UpdateUI("AR Session 未绑定，无法执行退出重置", Color.red);
            yield break;
        }

        int frameSequenceBeforeReset = cameraFrameSequence;
        hasCameraFrameSnapshot = false;
        cameraFrameHasCaptureAnchorPose = false;
        arSession.Reset();

        float deadline = Time.realtimeSinceStartup
            + Mathf.Max(1.0f, trackingResetTimeoutSeconds);
        float stableSince = -1.0f;
        bool freshCameraFrameObserved = false;
        while (Time.realtimeSinceStartup < deadline)
        {
            if (lifecycleGeneration != arSessionLifecycleGeneration)
                yield break;
            freshCameraFrameObserved =
                hasCameraFrameSnapshot
                && cameraFrameSequence > frameSequenceBeforeReset;
            bool stable =
                ARSession.state == ARSessionState.SessionTracking
                && freshCameraFrameObserved;
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

        arSessionLifecycleTransitionInProgress = false;
        recordButton.interactable = true;
        buttonText.text = "开始录制";
        buttonText.color = Color.white;
        bool resetReady =
            freshCameraFrameObserved
            && stableSince >= 0.0f
            && Time.realtimeSinceStartup - stableSince
                >= Mathf.Max(0.0f, trackingWarmupSeconds);
        UpdateUI(
            resetReady
                ? "已退出并完成 AR Session 硬重置，可开始全新录制"
                : "AR Session 已重置，但尚未恢复稳定 Tracking；请对准有纹理环境后再录制",
            resetReady ? Color.white : Color.yellow
        );
    }
    // ==========================================

    void Update()
    {
        if (fitStatusTextToPortraitSafeArea && Screen.safeArea != lastStatusSafeArea)
            ConfigureStatusTextForPortrait();
        if (
            useBottomSafeAreaMeshControlDock
            && (
                Screen.safeArea != lastMeshControlSafeArea
                || Screen.width != lastMeshControlScreenWidth
                || Screen.height != lastMeshControlScreenHeight
            )
        )
            ConfigureBottomActionDocksForSafeArea();
        RefreshPrimaryActionDockVisibility();
        UpdateCaptureReferenceAnchorTracking();
        UpdateReconstructedSilhouette(false);
        UpdateServerStyleOutlineCamera();
        UpdateMobileOverlayAudit();
        if (ARSession.state != ARSessionState.SessionTracking) return;

        if (
            arCamera != null
            && alignmentRefinementState == AlignmentRefinementState.Capturing
        )
        {
            if (
                !CaptureReferencePoseStable()
                || !TryGetPoseRelativeToCaptureAnchor(
                    cameraFramePosition,
                    cameraFrameRotation,
                    out Vector3 cameraA0Position,
                    out Quaternion cameraA0Rotation
                )
            )
            {
                UpdateUI(
                    "快速校准暂停：请对准原场景并等待 A0 稳定 Tracking",
                    Color.yellow
                );
                return;
            }
            UpdateUI(
                $"[快速校准采集中] 已上传 {alignmentRefinementUploadedCount} 帧；"
                + $"围绕物体中心缓慢移动，至少"
                + $"{Mathf.Max(16, alignmentRefineMinimumFrames)}帧、建议"
                + $"{Mathf.Max(32, alignmentRefineRecommendedFrames)}帧或更多；"
                + "服务端将球面FPS选16帧优化",
                Color.green
            );
            alignmentRefinementTimer += Time.deltaTime;
            if (
                alignmentRefinementTimer >= Mathf.Max(0.10f, alignmentRefineSendInterval)
                && !isSending
                && cameraManager != null
                && cameraFrameHasIntrinsics
                && cameraManager.TryAcquireLatestCpuImage(out XRCpuImage alignmentImage)
            )
            {
                alignmentRefinementTimer = 0.0f;
                long frameTimestampNs = hasCameraFrameSnapshot
                    ? cameraFrameTimestampNs
                    : -1;
                double cpuTimestampSeconds = alignmentImage.timestamp;
                double frameTimestampSeconds = frameTimestampNs > 0
                    ? frameTimestampNs * 1.0e-9
                    : -1.0;
                double timestampDeltaSeconds =
                    frameTimestampSeconds > 0.0 && cpuTimestampSeconds > 0.0
                        ? System.Math.Abs(
                            cpuTimestampSeconds - frameTimestampSeconds
                        )
                        : -1.0;
                StartCoroutine(
                    ProcessAndSendData(
                        alignmentImage,
                        cameraA0Position,
                        cameraA0Rotation,
                        cameraFrameHasIntrinsics,
                        cameraFrameIntrinsics,
                        cpuTimestampSeconds,
                        frameTimestampNs,
                        hasCameraFrameSnapshot
                            ? cameraFramePoseSampleSeconds
                            : (double)Time.realtimeSinceStartup,
                        timestampDeltaSeconds,
                        hasCameraFrameSnapshot
                            ? cameraFrameDisplayMatrix
                            : Matrix4x4.identity,
                        hasCameraFrameSnapshot
                            ? cameraFrameProjectionMatrix
                            : Matrix4x4.identity,
                        "camera_frame_received_anchor_a0_relative_refinement_v1",
                        Vector3.zero,
                        -1.0f,
                        "/alignment_refine/upload",
                        "unity_capture_anchor_a0",
                        activeServerSessionId,
                        activeAlignmentRefinementId,
                        arSessionLifecycleGeneration
                    )
                );
            }
            return;
        }

        if (arCamera != null && isRecording)
        {
            if (!cameraFrameHasCaptureAnchorPose || !CaptureReferencePoseUsable())
            {
                UpdateUI(
                    "A0 Anchor 暂未 Tracking：暂停上传，保留录制状态并等待重定位",
                    Color.yellow
                );
                return;
            }
            UpdateUI(
                $"[录制中] 固定0.2秒上传；客户端不筛帧；已上传 "
                    + $"{acceptedPoseDiversityDirections.Count} 帧\n"
                    + $"A0 Pos: {cameraFrameCaptureAnchorPosition.x:F2}, "
                    + $"{cameraFrameCaptureAnchorPosition.y:F2}, "
                    + $"{cameraFrameCaptureAnchorPosition.z:F2}",
                Color.green
            );

            timer += Time.deltaTime;
            if (timer >= sendInterval && !isSending)
            {
                timer = 0f;
                Vector3 framePosition = cameraFrameCaptureAnchorPosition;
                Quaternion frameRotation = cameraFrameCaptureAnchorRotation;
                MeasurePoseDiversityForAudit(
                    framePosition,
                    frameRotation,
                    out Vector3 diversityDirection,
                    out float diversityMinimumAngle
                );
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
                            "camera_frame_received_anchor_a0_relative_v1",
                            diversityDirection,
                            diversityMinimumAngle,
                            "/upload",
                            "unity_capture_anchor_a0",
                            activeServerSessionId,
                            "",
                            arSessionLifecycleGeneration
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
                if (
                    response == null
                    || string.IsNullOrEmpty(response.session_id)
                )
                {
                    UpdateUI("准备分割失败：服务端未返回会话标识", Color.red);
                    recordButton.gameObject.SetActive(true);
                    buttonText.text = "开始录制";
                    buttonText.color = Color.white;
                    yield break;
                }
                activeServerSessionId = response.session_id;
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

    void ConfigureUnifiedButtonStyle(
        Button button,
        Color normalColor,
        float minimumHeight,
        int minimumFontSize,
        int maximumFontSize
    )
    {
        if (button == null) return;
        ColorBlock colors = button.colors;
        colors.normalColor = normalColor;
        colors.highlightedColor = Color.Lerp(normalColor, Color.white, 0.16f);
        colors.pressedColor = Color.Lerp(normalColor, Color.black, 0.24f);
        colors.selectedColor = colors.highlightedColor;
        colors.disabledColor = new Color(
            normalColor.r,
            normalColor.g,
            normalColor.b,
            0.40f
        );
        colors.colorMultiplier = 1.0f;
        colors.fadeDuration = 0.08f;
        button.colors = colors;
        if (button.targetGraphic != null) button.targetGraphic.color = Color.white;

        LayoutElement layout = button.GetComponent<LayoutElement>();
        if (layout == null) layout = button.gameObject.AddComponent<LayoutElement>();
        layout.minHeight = minimumHeight;
        layout.preferredHeight = minimumHeight;

        // LayoutElement controls buttons below a LayoutGroup.  Some older
        // Unity scenes keep review buttons on absolute RectTransforms, so
        // enforce the same touch-target height there without requiring the
        // user to rebuild the Canvas or reconnect Inspector references.
        RectTransform buttonRect = button.GetComponent<RectTransform>();
        if (buttonRect != null && buttonRect.rect.height < minimumHeight)
            buttonRect.SetSizeWithCurrentAnchors(
                RectTransform.Axis.Vertical,
                minimumHeight
            );

        Text text = button.GetComponentInChildren<Text>();
        if (text == null) return;
        text.color = Color.white;
        text.fontStyle = FontStyle.Bold;
        text.alignment = TextAnchor.MiddleCenter;
        text.horizontalOverflow = HorizontalWrapMode.Wrap;
        text.verticalOverflow = VerticalWrapMode.Truncate;
        text.resizeTextForBestFit = true;
        text.resizeTextMinSize = Mathf.Max(12, minimumFontSize);
        text.resizeTextMaxSize = Mathf.Max(text.resizeTextMinSize, maximumFontSize);
        text.raycastTarget = false;
    }

    void ConfigureUnifiedReviewButtons()
    {
        ConfigureUnifiedButtonStyle(
            btnPrev, secondaryActionColor, 68.0f, 17, 27
        );
        ConfigureUnifiedButtonStyle(
            btnNext, secondaryActionColor, 68.0f, 17, 27
        );
        ConfigureUnifiedButtonStyle(
            btnToggleKeep, new Color(0.30f, 0.25f, 0.58f, 0.96f), 68.0f, 17, 27
        );
        ConfigureUnifiedButtonStyle(
            btnSubmit, primaryRecordColor, 76.0f, 18, 29
        );
        ConfigureUnifiedButtonStyle(
            btnCancel, destructiveActionColor, 68.0f, 17, 27
        );
    }

    void EnsurePrimaryActionDock()
    {
        if (
            !useBottomSafeAreaMeshControlDock
            || primaryActionDockObject != null
            || recordButton == null
        )
            return;
        Canvas canvas = recordButton.GetComponentInParent<Canvas>();
        if (canvas == null) return;
        if (canvas.rootCanvas != null) canvas = canvas.rootCanvas;

        primaryActionDockObject = new GameObject(
            "ARPrimaryRecordSafeAreaDock",
            typeof(RectTransform),
            typeof(CanvasRenderer),
            typeof(Image),
            typeof(HorizontalLayoutGroup)
        );
        primaryActionDockObject.transform.SetParent(canvas.transform, false);
        primaryActionDockRect = primaryActionDockObject.GetComponent<RectTransform>();
        Image background = primaryActionDockObject.GetComponent<Image>();
        background.color = new Color(0.02f, 0.04f, 0.07f, 0.80f);
        background.raycastTarget = false;
        HorizontalLayoutGroup layout =
            primaryActionDockObject.GetComponent<HorizontalLayoutGroup>();
        layout.padding = new RectOffset(12, 12, 10, 10);
        layout.spacing = 0.0f;
        layout.childAlignment = TextAnchor.MiddleCenter;
        layout.childControlWidth = true;
        layout.childControlHeight = true;
        layout.childForceExpandWidth = true;
        layout.childForceExpandHeight = true;

        recordButton.transform.SetParent(primaryActionDockRect, false);
        RectTransform rect = recordButton.GetComponent<RectTransform>();
        if (rect != null)
        {
            rect.anchorMin = new Vector2(0.5f, 0.5f);
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.pivot = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = Vector2.zero;
            rect.sizeDelta = new Vector2(0.0f, primaryRecordButtonHeight);
        }
        LayoutElement recordLayout = recordButton.GetComponent<LayoutElement>();
        if (recordLayout == null)
            recordLayout = recordButton.gameObject.AddComponent<LayoutElement>();
        recordLayout.minWidth = 280.0f;
        recordLayout.preferredWidth = 720.0f;
        recordLayout.flexibleWidth = 1.0f;
        recordLayout.minHeight = primaryRecordButtonHeight;
        recordLayout.preferredHeight = primaryRecordButtonHeight;
        ConfigureUnifiedButtonStyle(
            recordButton,
            primaryRecordColor,
            primaryRecordButtonHeight,
            primaryRecordMinFontSize,
            primaryRecordMaxFontSize
        );
        primaryActionDockRect.SetAsLastSibling();
    }

    void RefreshPrimaryActionDockVisibility()
    {
        if (primaryActionDockObject == null || recordButton == null) return;
        bool visible = recordButton.gameObject.activeSelf;
        if (primaryActionDockObject.activeSelf != visible)
            primaryActionDockObject.SetActive(visible);
        if (visible) ConfigureBottomActionDocksForSafeArea();
    }

    void EnsureMeshControlDock()
    {
        if (
            !useBottomSafeAreaMeshControlDock
            || meshControlDockObject != null
            || meshDisplayButton == null
        )
            return;

        Canvas canvas = meshDisplayButton.GetComponentInParent<Canvas>();
        if (canvas == null)
        {
            Debug.LogWarning(
                "[ARMeshUI] Mesh Display Button 不在 Canvas 下，保留原布局"
            );
            return;
        }
        if (canvas.rootCanvas != null) canvas = canvas.rootCanvas;

        meshControlDockObject = new GameObject(
            "ARMeshBottomSafeAreaControlDock",
            typeof(RectTransform),
            typeof(CanvasRenderer),
            typeof(Image),
            typeof(HorizontalLayoutGroup)
        );
        meshControlDockObject.transform.SetParent(canvas.transform, false);
        meshControlDockRect = meshControlDockObject.GetComponent<RectTransform>();
        Image background = meshControlDockObject.GetComponent<Image>();
        background.color = meshControlDockColor;
        background.raycastTarget = false;

        meshControlDockLayout =
            meshControlDockObject.GetComponent<HorizontalLayoutGroup>();
        meshControlDockLayout.padding = new RectOffset(12, 12, 10, 10);
        meshControlDockLayout.spacing = Mathf.Max(4.0f, meshControlButtonSpacing);
        meshControlDockLayout.childAlignment = TextAnchor.MiddleCenter;
        meshControlDockLayout.childControlWidth = true;
        meshControlDockLayout.childControlHeight = true;
        meshControlDockLayout.childForceExpandWidth = true;
        meshControlDockLayout.childForceExpandHeight = true;

        AttachMeshControlButton(
            meshDisplayButton,
            secondaryActionColor
        );
        AttachMeshControlButton(
            meshOutlineMethodButton,
            secondaryActionColor
        );
        AttachMeshControlButton(
            alignmentRefineButton,
            secondaryActionColor
        );
        AttachMeshControlButton(
            poseDiagnosticRecordButton,
            new Color(0.38f, 0.20f, 0.58f, 0.96f)
        );
        meshControlDockRect.SetAsLastSibling();
    }

    void AttachMeshControlButton(Button button, Color normalColor)
    {
        if (button == null || meshControlDockRect == null) return;
        RectTransform rect = button.GetComponent<RectTransform>();
        button.transform.SetParent(meshControlDockRect, false);
        button.transform.localScale = Vector3.one;
        button.transform.localRotation = Quaternion.identity;
        if (rect != null)
        {
            rect.anchorMin = new Vector2(0.5f, 0.5f);
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.pivot = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = Vector2.zero;
            rect.sizeDelta = new Vector2(0.0f, meshControlButtonHeight);
        }

        LayoutElement layout = button.GetComponent<LayoutElement>();
        if (layout == null) layout = button.gameObject.AddComponent<LayoutElement>();
        layout.minWidth = 170.0f;
        layout.preferredWidth = 280.0f;
        layout.flexibleWidth = 1.0f;
        layout.minHeight = Mathf.Max(72.0f, meshControlButtonHeight);
        layout.preferredHeight = Mathf.Max(72.0f, meshControlButtonHeight);
        layout.flexibleHeight = 0.0f;

        ColorBlock colors = button.colors;
        colors.normalColor = normalColor;
        colors.highlightedColor = Color.Lerp(normalColor, Color.white, 0.18f);
        colors.pressedColor = Color.Lerp(normalColor, Color.black, 0.22f);
        colors.selectedColor = colors.highlightedColor;
        colors.disabledColor = new Color(
            normalColor.r,
            normalColor.g,
            normalColor.b,
            0.42f
        );
        colors.colorMultiplier = 1.0f;
        colors.fadeDuration = 0.08f;
        button.colors = colors;
        if (button.targetGraphic != null)
            button.targetGraphic.color = Color.white;

        Text text = button.GetComponentInChildren<Text>();
        if (text != null)
        {
            text.color = Color.white;
            text.fontStyle = FontStyle.Bold;
            text.alignment = TextAnchor.MiddleCenter;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            text.resizeTextForBestFit = true;
            text.resizeTextMinSize = Mathf.Max(15, meshControlButtonMinFontSize);
            text.resizeTextMaxSize = Mathf.Max(
                text.resizeTextMinSize,
                meshControlButtonMaxFontSize
            );
            text.raycastTarget = false;
            RectTransform textRect = text.rectTransform;
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(10.0f, 6.0f);
            textRect.offsetMax = new Vector2(-10.0f, -6.0f);
        }
    }

    void ConfigureBottomActionDocksForSafeArea()
    {
        if (!useBottomSafeAreaMeshControlDock)
            return;
        lastMeshControlSafeArea = Screen.safeArea;
        lastMeshControlScreenWidth = Screen.width;
        lastMeshControlScreenHeight = Screen.height;
        if (Screen.width <= 0 || Screen.height <= 0) return;

        Rect safe = Screen.safeArea;
        float meshLeft = Mathf.Clamp01(
            safe.xMin / Screen.width + meshControlDockSideMarginRatio
        );
        float meshRight = Mathf.Clamp01(
            safe.xMax / Screen.width - meshControlDockSideMarginRatio
        );
        float primaryLeft = Mathf.Clamp01(
            safe.xMin / Screen.width + primaryRecordSideMarginRatio
        );
        float primaryRight = Mathf.Clamp01(
            safe.xMax / Screen.width - primaryRecordSideMarginRatio
        );
        float primaryBottom = Mathf.Clamp01(
            safe.yMin / Screen.height + primaryRecordBottomMarginRatio
        );
        if (meshRight <= meshLeft + 0.20f)
        {
            meshLeft = Mathf.Clamp01(safe.xMin / Screen.width + 0.01f);
            meshRight = Mathf.Clamp01(safe.xMax / Screen.width - 0.01f);
        }
        if (primaryRight <= primaryLeft + 0.20f)
        {
            primaryLeft = meshLeft;
            primaryRight = meshRight;
        }

        float primaryDockHeight = Mathf.Max(88.0f, primaryRecordButtonHeight) + 20.0f;
        if (primaryActionDockRect != null)
        {
            primaryActionDockRect.anchorMin = new Vector2(primaryLeft, primaryBottom);
            primaryActionDockRect.anchorMax = new Vector2(primaryRight, primaryBottom);
            primaryActionDockRect.pivot = new Vector2(0.5f, 0.0f);
            primaryActionDockRect.anchoredPosition = Vector2.zero;
            primaryActionDockRect.sizeDelta = new Vector2(0.0f, primaryDockHeight);
        }

        float meshBottom = Mathf.Clamp01(
            primaryBottom
            + (
                primaryDockHeight
                + Mathf.Max(4.0f, bottomDockVerticalSpacing)
            ) / Screen.height
            + meshControlDockBottomMarginRatio
        );
        float meshDockHeight = Mathf.Max(64.0f, meshControlButtonHeight) + 20.0f;
        if (meshControlDockRect != null)
        {
            meshControlDockRect.anchorMin = new Vector2(meshLeft, meshBottom);
            meshControlDockRect.anchorMax = new Vector2(meshRight, meshBottom);
            meshControlDockRect.pivot = new Vector2(0.5f, 0.0f);
            meshControlDockRect.anchoredPosition = Vector2.zero;
            meshControlDockRect.sizeDelta = new Vector2(0.0f, meshDockHeight);
        }
        if (meshControlDockLayout != null)
            meshControlDockLayout.spacing = Mathf.Max(4.0f, meshControlButtonSpacing);
        if (meshControlDockRect != null)
        {
            meshControlDockRect.SetAsLastSibling();
            LayoutRebuilder.ForceRebuildLayoutImmediate(meshControlDockRect);
        }
        if (primaryActionDockRect != null)
        {
            primaryActionDockRect.SetAsLastSibling();
            LayoutRebuilder.ForceRebuildLayoutImmediate(primaryActionDockRect);
        }
    }

    void RefreshMeshControlDockVisibility()
    {
        if (meshControlDockObject == null) return;
        bool anyVisible =
            (meshDisplayButton != null && meshDisplayButton.gameObject.activeSelf)
            || (
                meshOutlineMethodButton != null
                && meshOutlineMethodButton.gameObject.activeSelf
            )
            || (
                alignmentRefineButton != null
                && alignmentRefineButton.gameObject.activeSelf
            )
            || (
                poseDiagnosticRecordButton != null
                && poseDiagnosticRecordButton.gameObject.activeSelf
            );
        meshControlDockObject.SetActive(anyVisible);
        if (anyVisible) ConfigureBottomActionDocksForSafeArea();
    }

    void EnsureMeshOutlineMethodButton()
    {
        if (meshOutlineMethodButton == meshDisplayButton)
        {
            Debug.LogWarning(
                "[ARMeshOutline] display and method fields referenced the same Button; creating a separate runtime Button"
            );
            meshOutlineMethodButton = null;
        }
        if (meshOutlineMethodButton != null || meshDisplayButton == null) return;

        GameObject clone = Instantiate(
            meshDisplayButton.gameObject,
            meshDisplayButton.transform.parent
        );
        clone.name = "MeshOutlineMethodButton_Runtime";
        clone.transform.SetSiblingIndex(meshDisplayButton.transform.GetSiblingIndex() + 1);
        meshOutlineMethodButton = clone.GetComponent<Button>();
        if (meshOutlineMethodButton == null)
        {
            Destroy(clone);
            return;
        }
        meshOutlineMethodButton.onClick = new Button.ButtonClickedEvent();

        RectTransform sourceRect = meshDisplayButton.GetComponent<RectTransform>();
        RectTransform cloneRect = clone.GetComponent<RectTransform>();
        Transform parent = meshDisplayButton.transform.parent;
        bool parentControlsLayout =
            parent != null && parent.GetComponent<LayoutGroup>() != null;
        if (sourceRect != null && cloneRect != null && !parentControlsLayout)
        {
            float verticalOffset = Mathf.Max(8.0f, sourceRect.rect.height + 8.0f);
            cloneRect.anchoredPosition =
                sourceRect.anchoredPosition + Vector2.up * verticalOffset;
        }
        SetButtonLabel(meshOutlineMethodButton, "切换：屏幕轮廓");
    }

    void EnsureAlignmentRefineButton()
    {
        if (
            alignmentRefineButton == meshDisplayButton
            || alignmentRefineButton == meshOutlineMethodButton
        )
            alignmentRefineButton = null;
        if (alignmentRefineButton != null || meshDisplayButton == null) return;

        GameObject clone = Instantiate(
            meshDisplayButton.gameObject,
            meshDisplayButton.transform.parent
        );
        clone.name = "FastA0MeshAlignmentButton_Runtime";
        clone.transform.SetSiblingIndex(
            meshDisplayButton.transform.GetSiblingIndex() + 2
        );
        alignmentRefineButton = clone.GetComponent<Button>();
        if (alignmentRefineButton == null)
        {
            Destroy(clone);
            return;
        }
        alignmentRefineButton.onClick = new Button.ButtonClickedEvent();

        RectTransform sourceRect = meshDisplayButton.GetComponent<RectTransform>();
        RectTransform cloneRect = clone.GetComponent<RectTransform>();
        Transform parent = meshDisplayButton.transform.parent;
        bool parentControlsLayout =
            parent != null && parent.GetComponent<LayoutGroup>() != null;
        if (sourceRect != null && cloneRect != null && !parentControlsLayout)
        {
            float verticalOffset = Mathf.Max(8.0f, sourceRect.rect.height + 8.0f);
            cloneRect.anchoredPosition =
                sourceRect.anchoredPosition + Vector2.up * (2.0f * verticalOffset);
        }
        SetButtonLabel(alignmentRefineButton, "快速校准");
    }

    void EnsurePoseDiagnosticRecordButton()
    {
        if (
            poseDiagnosticRecordButton == meshDisplayButton
            || poseDiagnosticRecordButton == meshOutlineMethodButton
            || poseDiagnosticRecordButton == alignmentRefineButton
        )
            poseDiagnosticRecordButton = null;
        if (poseDiagnosticRecordButton != null || meshDisplayButton == null)
            return;

        GameObject clone = Instantiate(
            meshDisplayButton.gameObject,
            meshDisplayButton.transform.parent
        );
        clone.name = "PoseDiagnosticRecordButton_Runtime";
        clone.transform.SetSiblingIndex(
            meshDisplayButton.transform.GetSiblingIndex() + 3
        );
        poseDiagnosticRecordButton = clone.GetComponent<Button>();
        if (poseDiagnosticRecordButton == null)
        {
            Destroy(clone);
            return;
        }
        poseDiagnosticRecordButton.onClick = new Button.ButtonClickedEvent();

        RectTransform sourceRect = meshDisplayButton.GetComponent<RectTransform>();
        RectTransform cloneRect = clone.GetComponent<RectTransform>();
        Transform parent = meshDisplayButton.transform.parent;
        bool parentControlsLayout =
            parent != null && parent.GetComponent<LayoutGroup>() != null;
        if (sourceRect != null && cloneRect != null && !parentControlsLayout)
        {
            float verticalOffset = Mathf.Max(8.0f, sourceRect.rect.height + 8.0f);
            cloneRect.anchoredPosition =
                sourceRect.anchoredPosition + Vector2.up * (3.0f * verticalOffset);
        }
        SetButtonLabel(poseDiagnosticRecordButton, "录制位姿诊断");
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
        yield return StartCoroutine(CheckInputQcThenGenerate());
    }

    SelectionData BuildAllFrameRequest()
    {
        SelectionData data = new SelectionData();
        data.selected = new List<int>();
        for (int i = 0; i < capturedFrameCount; i++) data.selected.Add(i);
        data.session_id = activeServerSessionId;
        data.lifecycle_generation = arSessionLifecycleGeneration;
        return data;
    }

    IEnumerator CheckInputQcThenGenerate()
    {
        int lifecycleGeneration = arSessionLifecycleGeneration;
        string sessionId = activeServerSessionId;
        if (string.IsNullOrEmpty(sessionId))
        {
            UpdateUI("当前采集缺少服务端 session_id，拒绝继续", Color.red);
            yield break;
        }
        SelectionData data = BuildAllFrameRequest();

        string jsonPayload = JsonUtility.ToJson(data);
        UnityWebRequest qc = new UnityWebRequest(serverURL + "/input_qc", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        qc.uploadHandler = new UploadHandlerRaw(bodyRaw);
        qc.downloadHandler = new DownloadHandlerBuffer();
        qc.SetRequestHeader("Content-Type", "application/json");
        qc.timeout = 120;
        yield return qc.SendWebRequest();

        if (
            lifecycleGeneration != arSessionLifecycleGeneration
            || sessionId != activeServerSessionId
        )
            yield break;

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

        UpdateUI("输入检查通过，正在用全部 Pose+Mask 冻结 Runtime-O...", Color.yellow);
        yield return StartCoroutine(
            PrepareRuntimeOThenGenerate(data, lifecycleGeneration, sessionId)
        );
    }

    IEnumerator PrepareRuntimeOThenGenerate(
        SelectionData data,
        int lifecycleGeneration,
        string sessionId
    )
    {
        string prepareJson = JsonUtility.ToJson(data);
        UnityWebRequest prepare = new UnityWebRequest(
            serverURL + "/prepare_runtime_o",
            "POST"
        );
        prepare.uploadHandler = new UploadHandlerRaw(
            System.Text.Encoding.UTF8.GetBytes(prepareJson)
        );
        prepare.downloadHandler = new DownloadHandlerBuffer();
        prepare.SetRequestHeader("Content-Type", "application/json");
        prepare.timeout = 600;
        yield return prepare.SendWebRequest();

        if (
            lifecycleGeneration != arSessionLifecycleGeneration
            || sessionId != activeServerSessionId
        )
            yield break;
        if (prepare.result != UnityWebRequest.Result.Success)
        {
            UpdateUI("Runtime-O 准备失败: " + ExtractServerMessage(prepare), Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
            yield break;
        }

        RuntimeOPrepareResponse response = null;
        try
        {
            response = JsonUtility.FromJson<RuntimeOPrepareResponse>(
                prepare.downloadHandler.text
            );
        }
        catch
        {
            response = null;
        }
        if (
            response == null
            || response.status != "success"
            || response.session_id != sessionId
            || response.lifecycle_generation != lifecycleGeneration
            || string.IsNullOrEmpty(response.runtime_o_sha256)
            || string.IsNullOrEmpty(response.requested_pose_sha256)
            || response.model_inference_started
        )
        {
            UpdateUI("Runtime-O 响应合同无效，拒绝启动模型", Color.red);
            yield break;
        }

        preparedRuntimeOSha256 = response.runtime_o_sha256;
        preparedRequestedPoseSha256 = response.requested_pose_sha256;
        preparedLifecycleGeneration = lifecycleGeneration;
        data.runtime_o_sha256 = preparedRuntimeOSha256;
        data.requested_pose_sha256 = preparedRequestedPoseSha256;
        UpdateUI("Runtime-O 已冻结；服务器开始 SS30K+SLat30K 重建 Mesh...", Color.yellow);
        yield return StartCoroutine(
            SendGenerateCommand(
                JsonUtility.ToJson(data),
                lifecycleGeneration,
                sessionId
            )
        );
    }

    IEnumerator SendGenerateCommand(
        string json,
        int lifecycleGeneration,
        string sessionId
    )
    {
        UnityWebRequest www = new UnityWebRequest(serverURL + "/generate", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");

        www.timeout = 600;

        yield return www.SendWebRequest();

        if (
            lifecycleGeneration != arSessionLifecycleGeneration
            || sessionId != activeServerSessionId
        )
            yield break;

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
                && response.session_id == sessionId
                && response.mobile_ar != null
                && response.mobile_ar.session_id == sessionId
                && response.mobile_ar.lifecycle_generation == lifecycleGeneration
                && response.mobile_ar.runtime_o_sha256 == preparedRuntimeOSha256
                && response.mobile_ar.requested_pose_sha256
                    == preparedRequestedPoseSha256
                && !string.IsNullOrEmpty(response.mobile_ar.mesh_url);
            if (hasMobileMesh)
            {
                UpdateUI(
                    $"Mesh 重建完成（服务端 {response.mobile_ar.vertex_count} 顶点/"
                    + $"{response.mobile_ar.triangle_count} 三角形），正在加载...",
                    Color.yellow
                );
                yield return StartCoroutine(
                    DownloadAndDisplayReconstructedMesh(
                        response.mobile_ar,
                        lifecycleGeneration,
                        sessionId
                    )
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

    IEnumerator DownloadAndDisplayReconstructedMesh(
        MobileARResponse mobileAR,
        int lifecycleGeneration,
        string sessionId
    )
    {
        string meshUrl = mobileAR.mesh_url;
        string url = meshUrl.StartsWith("http://") || meshUrl.StartsWith("https://")
            ? meshUrl
            : serverURL.TrimEnd('/') + "/" + meshUrl.TrimStart('/');
        using (UnityWebRequest meshRequest = UnityWebRequest.Get(url))
        {
            meshRequest.timeout = 180;
            yield return meshRequest.SendWebRequest();
            if (
                lifecycleGeneration != arSessionLifecycleGeneration
                || sessionId != activeServerSessionId
            )
                yield break;
            if (meshRequest.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("AR Mesh 下载失败: " + ExtractServerMessage(meshRequest), Color.red);
                yield break;
            }
            byte[] downloadedBytes = meshRequest.downloadHandler.data;
            if (downloadedBytes == null || downloadedBytes.Length == 0)
            {
                UpdateUI("AR Mesh 下载成功但响应为空", Color.red);
                yield break;
            }
            if (mobileAR.byte_count > 0 && downloadedBytes.Length != mobileAR.byte_count)
            {
                UpdateUI(
                    $"AR Mesh 字节数不一致：收到 {downloadedBytes.Length}，"
                    + $"服务端声明 {mobileAR.byte_count}",
                    Color.red
                );
                yield break;
            }
            if (!string.IsNullOrEmpty(mobileAR.mesh_sha256))
            {
                string downloadedSha256 = ComputeSha256Hex(downloadedBytes);
                if (
                    !string.Equals(
                        downloadedSha256,
                        mobileAR.mesh_sha256,
                        System.StringComparison.OrdinalIgnoreCase
                    )
                )
                {
                    UpdateUI(
                        "AR Mesh SHA-256 不一致；下载内容已拒绝",
                        Color.red
                    );
                    yield break;
                }
            }
            Vector3[] vertices;
            Vector3[] normals;
            int[] triangles;
            try
            {
                ParseMobileARMesh(
                    downloadedBytes,
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
                ValidateMobileARA0Contract(mobileAR);
                ClearPendingReconstructedMesh();
                pendingReconstructedVertices = vertices;
                pendingReconstructedNormals = normals;
                pendingReconstructedTriangles = triangles;
                pendingMobileARResponse = mobileAR;
                bool meshAttached = TryFinalizeMeshUnderCaptureAnchor();
                Debug.Log(
                    $"[ARMesh] received_bytes={downloadedBytes.Length} "
                    + $"vertices={vertices.Length} triangles={triangles.Length / 3} "
                    + $"frame={mobileAR.coordinate_frame} "
                    + $"attached_to_a0={meshAttached}"
                );
                if (captureReferenceAnchorTrackingStable && meshAttached)
                {
                    UpdateUI(
                        $"重建完成：{vertices.Length} 顶点 / "
                        + $"{triangles.Length / 3} 三角形；Mesh 已直接绑定稳定 A0",
                        Color.green
                    );
                }
                else
                {
                    UpdateUI(
                        "Mesh 已下载并缓存，等待采集 A0 连续稳定 Tracking 后显示",
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

    static string ComputeSha256Hex(byte[] payload)
    {
        using (SHA256 algorithm = SHA256.Create())
        {
            byte[] digest = algorithm.ComputeHash(payload);
            StringBuilder result = new StringBuilder(digest.Length * 2);
            for (int i = 0; i < digest.Length; i++)
                result.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
            return result.ToString();
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

    static bool IsFinite(Quaternion value)
    {
        return
            !float.IsNaN(value.x) && !float.IsInfinity(value.x)
            && !float.IsNaN(value.y) && !float.IsInfinity(value.y)
            && !float.IsNaN(value.z) && !float.IsInfinity(value.z)
            && !float.IsNaN(value.w) && !float.IsInfinity(value.w);
    }

    void ValidateMobileARA0Contract(MobileARResponse mobileAR)
    {
        if (mobileAR == null)
            throw new System.InvalidOperationException("服务端没有返回 mobile_ar");
        if (mobileAR.format != "yxc_unity_ar_mesh.v1")
            throw new System.InvalidOperationException(
                "AR Mesh 格式不受支持: " + mobileAR.format
            );
        if (mobileAR.coordinate_frame != "unity_capture_anchor_a0")
            throw new System.InvalidOperationException(
                "AR Mesh 坐标不是 unity_capture_anchor_a0: "
                + mobileAR.coordinate_frame
            );
        if (mobileAR.placement != "capture_anchor_a0_direct")
            throw new System.InvalidOperationException(
                "AR Mesh 缺少直接绑定 capture A0 的合同: "
                + mobileAR.placement
            );
        if (
            mobileAR.session_id != activeServerSessionId
            || mobileAR.lifecycle_generation != arSessionLifecycleGeneration
            || mobileAR.lifecycle_generation != preparedLifecycleGeneration
            || mobileAR.runtime_o_sha256 != preparedRuntimeOSha256
            || mobileAR.requested_pose_sha256 != preparedRequestedPoseSha256
        )
            throw new System.InvalidOperationException(
                "AR Mesh 的 session/generation/Runtime-O/A0 绑定不属于当前轮次"
            );
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

        // A0 is the first camera-frame origin. It owns both capture history
        // and the final A0-local display Mesh.
        Vector3 requestedPosition = cameraFramePosition;
        // Preserve the initial Unity gravity/world axes.  Using the camera's
        // roll/pitch here would rotate gravity in every A0-relative pose and
        // violate the model's y-up input convention.
        Quaternion requestedRotation = Quaternion.identity;
        captureReferenceAnchor = null;
        try
        {
            captureReferenceAnchor = anchorManager.AddAnchor(
                new Pose(requestedPosition, requestedRotation)
            );
        }
        catch (System.Exception exception)
        {
            Debug.LogWarning("[ARAnchor] AddAnchor failed: " + exception.Message);
            captureReferenceAnchor = null;
        }

        if (captureReferenceAnchor == null)
        {
            Debug.LogWarning(
                "[ARAnchor] real ARAnchor creation failed; no frozen world-pose fallback is allowed"
            );
            return false;
        }

        captureReferenceUsesTrackedARAnchor = true;
        captureReferenceAnchorObject = captureReferenceAnchor.gameObject;
        captureReferenceAnchorObject.name = "CaptureReferenceARAnchor";
        captureReferenceAnchorPosition = captureReferenceAnchorObject.transform.position;
        captureReferenceAnchorRotation = captureReferenceAnchorObject.transform.rotation;
        float initialPositionDelta = Vector3.Distance(
            requestedPosition,
            captureReferenceAnchorPosition
        );
        float initialRotationDelta = Quaternion.Angle(
            requestedRotation,
            captureReferenceAnchorRotation
        );
        Debug.Log(
            $"[ARAnchor] capture-start A0 frozen from returned transform; "
            + $"request_delta={initialPositionDelta:F4}m/"
            + $"{initialRotationDelta:F2}deg "
            + $"tracking={captureReferenceAnchor.trackingState}"
        );

        captureReferenceAnchorPoseValid = true;
        captureReferenceAnchorTrackingStable = false;
        captureReferenceAnchorEverTracked =
            captureReferenceAnchor != null
            && captureReferenceAnchor.trackingState == TrackingState.Tracking;
        captureReferenceAnchorTrackingSince = -1.0f;
        lastCaptureReferenceAnchorTrackingState = captureReferenceAnchor != null
            ? captureReferenceAnchor.trackingState
            : TrackingState.Limited;
        return true;
    }

    bool CaptureReferencePoseUsable()
    {
        return
            !applicationPaused
            && ARSession.state == ARSessionState.SessionTracking
            && HasCaptureReferenceFrame()
            && captureReferenceAnchor.trackingState == TrackingState.Tracking;
    }

    bool TryGetPoseRelativeToCaptureAnchor(
        Vector3 worldPosition,
        Quaternion worldRotation,
        out Vector3 a0Position,
        out Quaternion a0Rotation
    )
    {
        a0Position = Vector3.zero;
        a0Rotation = Quaternion.identity;
        if (!CaptureReferencePoseUsable()) return false;
        Transform anchorTransform = captureReferenceAnchorObject.transform;
        a0Position = anchorTransform.InverseTransformPoint(worldPosition);
        a0Rotation = Quaternion.Inverse(anchorTransform.rotation) * worldRotation;
        return true;
    }

    bool HasCaptureReferenceFrame()
    {
        return
            captureReferenceAnchorPoseValid
            && captureReferenceUsesTrackedARAnchor
            && captureReferenceAnchor != null
            && captureReferenceAnchorObject != null;
    }

    bool CaptureReferencePoseStable()
    {
        return CaptureReferencePoseUsable() && captureReferenceAnchorTrackingStable;
    }

    bool TryFinalizeMeshUnderCaptureAnchor()
    {
        if (
            pendingMobileARResponse == null
            || pendingReconstructedVertices == null
            || pendingReconstructedTriangles == null
            || !CaptureReferencePoseStable()
        )
            return false;

        MobileARResponse completedMobileAR = pendingMobileARResponse;
        ValidateMobileARA0Contract(completedMobileAR);
        CreateReconstructedMeshOverlay(
            pendingReconstructedVertices,
            pendingReconstructedNormals,
            pendingReconstructedTriangles
        );
        ClearPendingReconstructedMesh();
        activeMobileARResponse = completedMobileAR;
        alignmentRefinementState = AlignmentRefinementState.Ready;
        activeAlignmentRefinementId = "";
        alignmentRefinementUploadedCount = 0;
        alignmentRefinementTimer = 0.0f;
        lastAlignmentRefinementAccepted = false;
        lastAlignmentRefinementReport = "";
        if (alignmentRefineButton != null)
        {
            alignmentRefineButton.interactable = true;
            alignmentRefineButton.gameObject.SetActive(true);
            SetButtonLabel(alignmentRefineButton, "快速校准");
        }
        if (poseDiagnosticRecordButton != null)
        {
            poseDiagnosticRecordButton.interactable = enableMobileOverlayAudit;
            poseDiagnosticRecordButton.gameObject.SetActive(true);
            SetButtonLabel(poseDiagnosticRecordButton, "录制位姿诊断");
        }
        RefreshMeshControlDockVisibility();
        ResetMobileOverlayAudit(autoStartMobileOverlayAudit);
        return true;
    }

    void UpdateCaptureReferenceAnchorTracking()
    {
        if (!captureReferenceAnchorPoseValid && pendingMobileARResponse == null)
            return;

        bool sessionTracking =
            !applicationPaused
            && ARSession.state == ARSessionState.SessionTracking;
        bool referenceAvailable = HasCaptureReferenceFrame();
        TrackingState a0State = referenceAvailable
            ? captureReferenceAnchor.trackingState
            : TrackingState.None;
        TrackingState effectiveA0State =
            referenceAvailable && !sessionTracking
                ? TrackingState.Limited
                : a0State;
        bool referencePoseUsable =
            referenceAvailable
            && sessionTracking
            && a0State == TrackingState.Tracking;

        bool wasStable = captureReferenceAnchorTrackingStable;
        if (referencePoseUsable)
        {
            captureReferenceAnchorEverTracked = true;
            if (captureReferenceAnchorTrackingSince < 0.0f)
                captureReferenceAnchorTrackingSince = Time.realtimeSinceStartup;
            captureReferenceAnchorTrackingStable =
                Time.realtimeSinceStartup - captureReferenceAnchorTrackingSince
                >= Mathf.Max(0.0f, reconstructedAnchorRecoveryStableSeconds);
        }
        else
        {
            captureReferenceAnchorTrackingSince = -1.0f;
            captureReferenceAnchorTrackingStable = false;
        }

        bool a0StateChanged =
            effectiveA0State != lastCaptureReferenceAnchorTrackingState;
        if (pendingMobileARResponse != null && captureReferenceAnchorTrackingStable)
        {
            try
            {
                TryFinalizeMeshUnderCaptureAnchor();
            }
            catch (System.Exception exception)
            {
                ClearReconstructedMesh();
                ClearPendingReconstructedMesh();
                UpdateUI("Mesh 直接绑定 A0 失败: " + exception.Message, Color.red);
            }
        }
        if (a0StateChanged || wasStable != captureReferenceAnchorTrackingStable)
            ApplyReconstructedMeshDisplayMode();
        if (
            a0StateChanged
            && captureReferenceAnchorEverTracked
            && !referencePoseUsable
        )
        {
            UpdateUI(
                $"采集参考 A0 当前为 {effectiveA0State}，Mesh 已隐藏；请对准原场景等待 Tracking",
                Color.yellow
            );
        }
        else if (
            reconstructedMeshRoot != null
            && !wasStable
            && captureReferenceAnchorTrackingStable
        )
        {
            UpdateUI("采集参考 A0 已恢复并稳定 Tracking，Mesh 已显示", Color.green);
        }
        lastCaptureReferenceAnchorTrackingState = effectiveA0State;
    }

    void ClearPendingReconstructedMesh()
    {
        pendingReconstructedVertices = null;
        pendingReconstructedNormals = null;
        pendingReconstructedTriangles = null;
        pendingMobileARResponse = null;
    }

    void ClearCaptureReferenceAnchor()
    {
        if (captureReferenceAnchorObject != null)
            Destroy(captureReferenceAnchorObject);
        captureReferenceAnchorObject = null;
        captureReferenceAnchor = null;
        captureReferenceUsesTrackedARAnchor = false;
        captureReferenceAnchorPosition = Vector3.zero;
        captureReferenceAnchorRotation = Quaternion.identity;
        captureReferenceAnchorPoseValid = false;
        captureReferenceAnchorTrackingStable = false;
        captureReferenceAnchorEverTracked = false;
        captureReferenceAnchorTrackingSince = -1.0f;
        lastCaptureReferenceAnchorTrackingState = TrackingState.None;
        cameraFrameHasCaptureAnchorPose = false;
    }

    void CreateReconstructedMeshOverlay(
        Vector3[] vertices,
        Vector3[] _normals,
        int[] triangles
    )
    {
        ClearReconstructedMesh();
        if (!HasCaptureReferenceFrame())
            throw new System.InvalidOperationException("采集参考 A0 Anchor 不可用");

        reconstructedMeshRoot = new GameObject("ReconstructedObject_A0Local_Under_A0");
        reconstructedMeshRoot.transform.SetParent(
            captureReferenceAnchorObject.transform,
            false
        );
        // The service has already converted every vertex into the exact A0
        // local frame bound in /generate. Never apply T_O2A0 again in Unity.
        reconstructedMeshRoot.transform.localPosition = Vector3.zero;
        reconstructedMeshRoot.transform.localRotation = Quaternion.identity;
        reconstructedMeshRoot.transform.localScale = Vector3.one;

        reconstructedOutlineObject = new GameObject("ViewDependentSilhouette");
        reconstructedOutlineObject.transform.SetParent(reconstructedMeshRoot.transform, false);
        MeshFilter outlineFilter = reconstructedOutlineObject.AddComponent<MeshFilter>();
        MeshRenderer outlineRenderer = reconstructedOutlineObject.AddComponent<MeshRenderer>();
        reconstructedOutlineMesh = new Mesh();
        reconstructedOutlineMesh.name = "ReconstructedObjectViewDependentSilhouette";
        reconstructedOutlineMesh.indexFormat = vertices.Length > 65535
            ? IndexFormat.UInt32
            : IndexFormat.UInt16;
        reconstructedOutlineMesh.vertices = vertices;
        reconstructedOutlineVertices = vertices;
        reconstructedOutlineTriangles = triangles;
        reconstructedSilhouetteEdges = BuildSilhouetteTopology(triangles);
        reconstructedTriangleFrontFacing = new bool[triangles.Length / 3];
        outlineFilter.sharedMesh = reconstructedOutlineMesh;
        reconstructedOutlineMaterial = CreateOutlineMaterial(
            reconstructedOutlineMaterialTemplate,
            reconstructedOutlineColor
        );
        outlineRenderer.sharedMaterial = reconstructedOutlineMaterial;
        outlineRenderer.shadowCastingMode = ShadowCastingMode.Off;
        outlineRenderer.receiveShadows = false;

        // Keep the default 3D-line path free of the duplicate mask Mesh and
        // RenderTexture allocation.  Server-style GPU resources are created
        // lazily on the first method switch.
        reconstructedServerStyleAvailable = CanUseServerStyleOutline();
        reconstructedMeshDisplayMode = 0;
        reconstructedOutlineMethod = ReconstructedOutlineMethod.ViewDependentMeshLines;
        nextReconstructedSilhouetteUpdateSeconds = -1.0f;
        UpdateReconstructedSilhouette(true);
        ApplyReconstructedMeshDisplayMode();
        if (meshDisplayButton != null) meshDisplayButton.gameObject.SetActive(true);
        if (meshOutlineMethodButton != null)
            meshOutlineMethodButton.gameObject.SetActive(
                reconstructedServerStyleAvailable
            );
        RefreshMeshControlDockVisibility();
    }

    static List<SilhouetteEdge> BuildSilhouetteTopology(int[] triangles)
    {
        Dictionary<ulong, SilhouetteEdge> byKey =
            new Dictionary<ulong, SilhouetteEdge>(triangles.Length);
        List<SilhouetteEdge> edges = new List<SilhouetteEdge>(triangles.Length);
        for (int i = 0; i + 2 < triangles.Length; i += 3)
        {
            int face = i / 3;
            AddSilhouetteEdge(triangles[i], triangles[i + 1], face, byKey, edges);
            AddSilhouetteEdge(triangles[i + 1], triangles[i + 2], face, byKey, edges);
            AddSilhouetteEdge(triangles[i + 2], triangles[i], face, byKey, edges);
        }
        return edges;
    }

    static void AddSilhouetteEdge(
        int first,
        int second,
        int face,
        Dictionary<ulong, SilhouetteEdge> byKey,
        List<SilhouetteEdge> edges
    )
    {
        uint low = (uint)Mathf.Min(first, second);
        uint high = (uint)Mathf.Max(first, second);
        ulong key = ((ulong)low << 32) | high;
        if (byKey.TryGetValue(key, out SilhouetteEdge existing))
        {
            // A regular triangle mesh has at most two incident faces.  For a
            // non-manifold edge, retaining the latest second face still yields
            // a conservative visible contour instead of drawing every edge.
            existing.secondFace = face;
            return;
        }
        SilhouetteEdge edge = new SilhouetteEdge
        {
            firstVertex = (int)low,
            secondVertex = (int)high,
            firstFace = face,
        };
        byKey.Add(key, edge);
        edges.Add(edge);
    }

    void UpdateReconstructedSilhouette(bool force)
    {
        if (
            reconstructedOutlineMesh == null
            || reconstructedOutlineObject == null
            || reconstructedOutlineVertices == null
            || reconstructedOutlineTriangles == null
            || reconstructedSilhouetteEdges == null
            || reconstructedTriangleFrontFacing == null
            || arCamera == null
        )
            return;
        if (!force)
        {
            if (
                !captureReferenceAnchorTrackingStable
                || reconstructedMeshDisplayMode != 0
                || reconstructedOutlineMethod
                    != ReconstructedOutlineMethod.ViewDependentMeshLines
            )
                return;
            if (Time.realtimeSinceStartup < nextReconstructedSilhouetteUpdateSeconds)
                return;
        }
        nextReconstructedSilhouetteUpdateSeconds =
            Time.realtimeSinceStartup
            + Mathf.Max(0.02f, reconstructedSilhouetteUpdateIntervalSeconds);

        Vector3 cameraLocal = reconstructedOutlineObject.transform.InverseTransformPoint(
            arCamera.position
        );
        int faceCount = reconstructedOutlineTriangles.Length / 3;
        for (int face = 0; face < faceCount; face++)
        {
            int offset = face * 3;
            Vector3 first = reconstructedOutlineVertices[
                reconstructedOutlineTriangles[offset]
            ];
            Vector3 second = reconstructedOutlineVertices[
                reconstructedOutlineTriangles[offset + 1]
            ];
            Vector3 third = reconstructedOutlineVertices[
                reconstructedOutlineTriangles[offset + 2]
            ];
            Vector3 normal = Vector3.Cross(second - first, third - first);
            Vector3 center = (first + second + third) / 3.0f;
            reconstructedTriangleFrontFacing[face] =
                normal.sqrMagnitude > 1.0e-16f
                && Vector3.Dot(normal, cameraLocal - center) >= 0.0f;
        }

        reconstructedSilhouetteLineIndices.Clear();
        foreach (SilhouetteEdge edge in reconstructedSilhouetteEdges)
        {
            bool firstFront = reconstructedTriangleFrontFacing[edge.firstFace];
            bool isSilhouette = edge.secondFace < 0
                ? firstFront
                : firstFront
                    != reconstructedTriangleFrontFacing[edge.secondFace];
            if (!isSilhouette) continue;
            reconstructedSilhouetteLineIndices.Add(edge.firstVertex);
            reconstructedSilhouetteLineIndices.Add(edge.secondVertex);
        }
        reconstructedOutlineMesh.SetIndices(
            reconstructedSilhouetteLineIndices,
            MeshTopology.Lines,
            0,
            true
        );
    }

    static Material CreateOutlineMaterial(
        Material template,
        Color color
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
                throw new System.InvalidOperationException(
                    "请在 ARPoseTracker Inspector 中绑定 Reconstructed Outline Material Template"
                );
            }
            material = new Material(shader);
        }
        material.name = "ARMeshViewDependentSilhouette";
        material.color = color;
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        return material;
    }

    bool TryCreateServerStyleOutlineResources(
        Vector3[] vertices,
        int[] triangles
    )
    {
        reconstructedARRenderCamera = arCamera != null
            ? arCamera.GetComponent<Camera>()
            : null;
        if (reconstructedARRenderCamera == null)
            reconstructedARRenderCamera = Camera.main;
        if (reconstructedARRenderCamera == null)
        {
            Debug.LogWarning(
                "[ARMeshOutline] server-style mode unavailable: AR Camera component missing"
            );
            return false;
        }

        Shader maskShader = serverStyleMaskMaterialTemplate != null
            ? serverStyleMaskMaterialTemplate.shader
            : Shader.Find("Tracker/ARMeshSilhouetteMask");
        Shader outlineShader = serverStyleOutlineMaterialTemplate != null
            ? serverStyleOutlineMaterialTemplate.shader
            : Shader.Find("Tracker/ARMeshScreenSpaceOutline");
        if (maskShader == null || outlineShader == null)
        {
            Debug.LogWarning(
                "[ARMeshOutline] server-style mode unavailable: bind both server-style Materials in Inspector"
            );
            return false;
        }

        try
        {
            int layer = Mathf.Clamp(serverStyleOutlineLayer, 0, 31);
            int layerMask = 1 << layer;
            reconstructedMainCameraOriginalCullingMask =
                reconstructedARRenderCamera.cullingMask;
            reconstructedMainCameraCullingMaskCaptured = true;
            reconstructedARRenderCamera.cullingMask &= ~layerMask;

            reconstructedServerMaskObject = new GameObject(
                "ServerStyleFilledSilhouetteMask"
            );
            reconstructedServerMaskObject.layer = layer;
            reconstructedServerMaskObject.transform.SetParent(
                reconstructedMeshRoot.transform,
                false
            );
            MeshFilter maskFilter =
                reconstructedServerMaskObject.AddComponent<MeshFilter>();
            MeshRenderer maskRenderer =
                reconstructedServerMaskObject.AddComponent<MeshRenderer>();
            reconstructedServerMaskMesh = new Mesh();
            reconstructedServerMaskMesh.name = "ReconstructedObjectFilledSilhouette";
            reconstructedServerMaskMesh.indexFormat = vertices.Length > 65535
                ? IndexFormat.UInt32
                : IndexFormat.UInt16;
            reconstructedServerMaskMesh.vertices = vertices;
            reconstructedServerMaskMesh.triangles = triangles;
            reconstructedServerMaskMesh.RecalculateBounds();
            maskFilter.sharedMesh = reconstructedServerMaskMesh;
            reconstructedServerMaskMaterial = serverStyleMaskMaterialTemplate != null
                ? new Material(serverStyleMaskMaterialTemplate)
                : new Material(maskShader);
            reconstructedServerMaskMaterial.name = "ARMeshServerStyleMask";
            if (reconstructedServerMaskMaterial.HasProperty("_BaseColor"))
                reconstructedServerMaskMaterial.SetColor("_BaseColor", Color.white);
            if (reconstructedServerMaskMaterial.HasProperty("_Color"))
                reconstructedServerMaskMaterial.SetColor("_Color", Color.white);
            maskRenderer.sharedMaterial = reconstructedServerMaskMaterial;
            maskRenderer.shadowCastingMode = ShadowCastingMode.Off;
            maskRenderer.receiveShadows = false;

            reconstructedServerMaskCameraObject = new GameObject(
                "ARMeshServerStyleMaskCamera"
            );
            reconstructedServerMaskCameraObject.transform.SetParent(
                reconstructedARRenderCamera.transform,
                false
            );
            reconstructedServerMaskCamera =
                reconstructedServerMaskCameraObject.AddComponent<Camera>();
            reconstructedServerMaskCamera.CopyFrom(reconstructedARRenderCamera);
            reconstructedServerMaskCamera.clearFlags = CameraClearFlags.SolidColor;
            reconstructedServerMaskCamera.backgroundColor = new Color(0, 0, 0, 0);
            reconstructedServerMaskCamera.cullingMask = layerMask;
            reconstructedServerMaskCamera.allowHDR = false;
            reconstructedServerMaskCamera.allowMSAA = false;
            reconstructedServerMaskCamera.depthTextureMode = DepthTextureMode.None;
            reconstructedServerMaskCamera.stereoTargetEye = StereoTargetEyeMask.None;

            reconstructedServerOutlineCanvasObject = new GameObject(
                "ARMeshServerStyleScreenOutlineCanvas"
            );
            Canvas outlineCanvas =
                reconstructedServerOutlineCanvasObject.AddComponent<Canvas>();
            outlineCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
            outlineCanvas.overrideSorting = true;
            outlineCanvas.sortingOrder = -100;
            CanvasGroup canvasGroup =
                reconstructedServerOutlineCanvasObject.AddComponent<CanvasGroup>();
            canvasGroup.blocksRaycasts = false;
            canvasGroup.interactable = false;

            GameObject imageObject = new GameObject(
                "ServerStyleCyanOutsideBoundary",
                typeof(RectTransform)
            );
            imageObject.transform.SetParent(
                reconstructedServerOutlineCanvasObject.transform,
                false
            );
            RectTransform imageRect = imageObject.GetComponent<RectTransform>();
            imageRect.anchorMin = Vector2.zero;
            imageRect.anchorMax = Vector2.one;
            imageRect.offsetMin = Vector2.zero;
            imageRect.offsetMax = Vector2.zero;
            reconstructedServerOutlineImage = imageObject.AddComponent<RawImage>();
            reconstructedServerOutlineImage.raycastTarget = false;
            reconstructedServerOutlineImage.color = Color.white;
            reconstructedServerOutlineMaterial =
                serverStyleOutlineMaterialTemplate != null
                    ? new Material(serverStyleOutlineMaterialTemplate)
                    : new Material(outlineShader);
            reconstructedServerOutlineMaterial.name =
                "ARMeshServerStyleScreenBoundary";
            if (reconstructedServerOutlineMaterial.HasProperty("_OutlineColor"))
                reconstructedServerOutlineMaterial.SetColor(
                    "_OutlineColor",
                    reconstructedOutlineColor
                );
            reconstructedServerOutlineImage.material =
                reconstructedServerOutlineMaterial;

            EnsureServerStyleRenderTexture();
            reconstructedServerMaskObject.SetActive(false);
            reconstructedServerMaskCamera.enabled = false;
            reconstructedServerMaskCameraObject.SetActive(false);
            reconstructedServerOutlineCanvasObject.SetActive(false);
            return true;
        }
        catch (System.Exception exception)
        {
            Debug.LogWarning(
                "[ARMeshOutline] failed to create server-style resources: "
                + exception.Message
            );
            ClearServerStyleOutlineResources();
            return false;
        }
    }

    bool CanUseServerStyleOutline()
    {
        Camera renderCamera = arCamera != null
            ? arCamera.GetComponent<Camera>()
            : Camera.main;
        if (renderCamera == null) return false;
        Shader maskShader = serverStyleMaskMaterialTemplate != null
            ? serverStyleMaskMaterialTemplate.shader
            : Shader.Find("Tracker/ARMeshSilhouetteMask");
        Shader outlineShader = serverStyleOutlineMaterialTemplate != null
            ? serverStyleOutlineMaterialTemplate.shader
            : Shader.Find("Tracker/ARMeshScreenSpaceOutline");
        return maskShader != null && outlineShader != null;
    }

    void EnsureServerStyleRenderTexture()
    {
        if (
            reconstructedServerMaskCamera == null
            || reconstructedServerOutlineImage == null
        )
            return;
        float scale = Mathf.Clamp(serverStyleRenderScale, 0.25f, 1.0f);
        int width = Mathf.Max(64, Mathf.RoundToInt(Screen.width * scale));
        int height = Mathf.Max(64, Mathf.RoundToInt(Screen.height * scale));
        if (
            reconstructedServerMaskTexture != null
            && width == reconstructedServerMaskTextureWidth
            && height == reconstructedServerMaskTextureHeight
        )
            return;

        if (reconstructedServerMaskTexture != null)
        {
            reconstructedServerMaskCamera.targetTexture = null;
            reconstructedServerOutlineImage.texture = null;
            reconstructedServerMaskTexture.Release();
            Destroy(reconstructedServerMaskTexture);
        }
        reconstructedServerMaskTexture = new RenderTexture(
            width,
            height,
            24,
            RenderTextureFormat.ARGB32
        );
        reconstructedServerMaskTexture.name = "ARMeshServerStyleSilhouetteRT";
        reconstructedServerMaskTexture.filterMode = FilterMode.Bilinear;
        reconstructedServerMaskTexture.wrapMode = TextureWrapMode.Clamp;
        reconstructedServerMaskTexture.useMipMap = false;
        reconstructedServerMaskTexture.autoGenerateMips = false;
        reconstructedServerMaskTexture.Create();
        reconstructedServerMaskTextureWidth = width;
        reconstructedServerMaskTextureHeight = height;
        reconstructedServerMaskCamera.targetTexture = reconstructedServerMaskTexture;
        reconstructedServerOutlineImage.texture = reconstructedServerMaskTexture;
        if (
            reconstructedServerOutlineMaterial != null
            && reconstructedServerOutlineMaterial.HasProperty("_OutlineWidthPixels")
        )
        {
            reconstructedServerOutlineMaterial.SetFloat(
                "_OutlineWidthPixels",
                Mathf.Max(1.0f, serverStyleOutlineWidthPixels * scale)
            );
        }
    }

    void UpdateServerStyleOutlineCamera()
    {
        if (
            !reconstructedServerStyleAvailable
            || reconstructedServerMaskCamera == null
            || reconstructedARRenderCamera == null
        )
            return;
        bool shouldRender =
            captureReferenceAnchorTrackingStable
            && reconstructedMeshDisplayMode == 0
            && reconstructedOutlineMethod
                == ReconstructedOutlineMethod.ServerStyleScreenSpace;
        if (!shouldRender)
        {
            reconstructedServerMaskCamera.enabled = false;
            return;
        }

        float now = Time.realtimeSinceStartup;
        if (now < nextServerStyleRenderSeconds)
        {
            reconstructedServerMaskCamera.enabled = false;
            return;
        }
        nextServerStyleRenderSeconds = now
            + 1.0f / Mathf.Clamp(serverStyleMaxFramesPerSecond, 5.0f, 60.0f);

        EnsureServerStyleRenderTexture();
        reconstructedServerMaskCamera.transform.SetPositionAndRotation(
            reconstructedARRenderCamera.transform.position,
            reconstructedARRenderCamera.transform.rotation
        );
        reconstructedServerMaskCamera.nearClipPlane =
            reconstructedARRenderCamera.nearClipPlane;
        reconstructedServerMaskCamera.farClipPlane =
            reconstructedARRenderCamera.farClipPlane;
        reconstructedServerMaskCamera.orthographic =
            reconstructedARRenderCamera.orthographic;
        reconstructedServerMaskCamera.orthographicSize =
            reconstructedARRenderCamera.orthographicSize;
        reconstructedServerMaskCamera.projectionMatrix =
            reconstructedARRenderCamera.projectionMatrix;
        // Cameras render after Update/LateUpdate.  Enabling it only on due
        // frames caps the expensive mask pass while the RawImage keeps the
        // previous RenderTexture visible between updates.
        reconstructedServerMaskCamera.enabled = true;
    }

    void ClearServerStyleOutlineResources()
    {
        if (
            reconstructedARRenderCamera != null
            && reconstructedMainCameraCullingMaskCaptured
        )
        {
            int layer = Mathf.Clamp(serverStyleOutlineLayer, 0, 31);
            int layerMask = 1 << layer;
            if ((reconstructedMainCameraOriginalCullingMask & layerMask) != 0)
                reconstructedARRenderCamera.cullingMask |= layerMask;
            else
                reconstructedARRenderCamera.cullingMask &= ~layerMask;
        }
        if (reconstructedServerMaskCamera != null)
            reconstructedServerMaskCamera.targetTexture = null;
        if (reconstructedServerOutlineImage != null)
            reconstructedServerOutlineImage.texture = null;
        if (reconstructedServerMaskTexture != null)
        {
            reconstructedServerMaskTexture.Release();
            Destroy(reconstructedServerMaskTexture);
        }
        if (reconstructedServerMaskCameraObject != null)
            Destroy(reconstructedServerMaskCameraObject);
        if (reconstructedServerOutlineCanvasObject != null)
            Destroy(reconstructedServerOutlineCanvasObject);
        if (reconstructedServerMaskObject != null)
            Destroy(reconstructedServerMaskObject);
        if (reconstructedServerMaskMesh != null)
            Destroy(reconstructedServerMaskMesh);
        if (reconstructedServerMaskMaterial != null)
            Destroy(reconstructedServerMaskMaterial);
        if (reconstructedServerOutlineMaterial != null)
            Destroy(reconstructedServerOutlineMaterial);

        reconstructedServerMaskObject = null;
        reconstructedServerMaskMesh = null;
        reconstructedServerMaskMaterial = null;
        reconstructedServerMaskCameraObject = null;
        reconstructedServerMaskCamera = null;
        reconstructedServerMaskTexture = null;
        reconstructedServerOutlineCanvasObject = null;
        reconstructedServerOutlineImage = null;
        reconstructedServerOutlineMaterial = null;
        reconstructedARRenderCamera = null;
        reconstructedServerMaskTextureWidth = 0;
        reconstructedServerMaskTextureHeight = 0;
        reconstructedMainCameraCullingMaskCaptured = false;
        reconstructedServerStyleAvailable = false;
        nextServerStyleRenderSeconds = -1.0f;
    }

    MeshTransformUnity CurrentReconstructedMeshTransform()
    {
        Transform value = reconstructedMeshRoot != null
            ? reconstructedMeshRoot.transform
            : null;
        Vector3 position = value != null ? value.localPosition : Vector3.zero;
        Quaternion rotation = value != null
            ? value.localRotation
            : Quaternion.identity;
        Vector3 scale = value != null ? value.localScale : Vector3.one;
        float uniformScale = (scale.x + scale.y + scale.z) / 3.0f;
        return new MeshTransformUnity
        {
            position_x = position.x,
            position_y = position.y,
            position_z = position.z,
            quaternion_x = rotation.x,
            quaternion_y = rotation.y,
            quaternion_z = rotation.z,
            quaternion_w = rotation.w,
            uniform_scale = uniformScale,
        };
    }

    string CurrentPoseDiagnosticStage()
    {
        if (!string.IsNullOrEmpty(lastAlignmentRefinementReport))
            return lastAlignmentRefinementAccepted
                ? "post_fast_alignment_accepted"
                : "post_fast_alignment_rejected";
        return "pre_fast_alignment";
    }

    public void StartExplicitPoseDiagnosticRecording()
    {
        if (!enableMobileOverlayAudit)
        {
            UpdateUI("位姿诊断录制已在 Inspector 中关闭", Color.yellow);
            return;
        }
        if (
            mobileOverlayAuditStartPending
            || mobileOverlayAuditStartInFlight
            || mobileOverlayAuditCaptureActive
            || mobileOverlayAuditSending
        )
        {
            UpdateUI(
                $"位姿诊断正在录制：{mobileOverlayAuditUploadedCount}/"
                    + $"{Mathf.Max(1, mobileOverlayAuditServerMaximumFrames)}",
                Color.yellow
            );
            return;
        }
        if (
            reconstructedMeshRoot == null
            || activeMobileARResponse == null
            || !CaptureReferencePoseStable()
        )
        {
            UpdateUI(
                "位姿诊断需要已显示的 Mesh 和稳定 Tracking 的 A0",
                Color.yellow
            );
            return;
        }
        if (
            alignmentRefinementState == AlignmentRefinementState.Capturing
            || alignmentRefinementState == AlignmentRefinementState.Optimizing
        )
        {
            UpdateUI("请先结束当前快速校准，再录制位姿诊断", Color.yellow);
            return;
        }
        if (reconstructedMeshDisplayMode != 0)
        {
            UpdateUI(
                "请先切换到 Mesh 轮廓显示，再开始严格同位姿诊断",
                Color.yellow
            );
            return;
        }
        string stage = CurrentPoseDiagnosticStage();
        ResetMobileOverlayAudit(true, stage);
        UpdateUI(
            stage.StartsWith("post_")
                ? "正在准备校准后的位姿诊断；请围绕物体缓慢移动"
                : "正在准备校准前的位姿诊断；请围绕物体缓慢移动",
            Color.green
        );
    }

    void ResetMobileOverlayAudit(
        bool startWhenEligible,
        string diagnosticStage = ""
    )
    {
        mobileOverlayAuditGeneration++;
        mobileOverlayAuditDiagnosticStage = string.IsNullOrEmpty(diagnosticStage)
            ? CurrentPoseDiagnosticStage()
            : diagnosticStage;
        mobileOverlayAuditStartPending =
            startWhenEligible && enableMobileOverlayAudit;
        mobileOverlayAuditStartInFlight = false;
        mobileOverlayAuditCaptureActive = false;
        mobileOverlayAuditSending = false;
        activeMobileOverlayAuditId = "";
        mobileOverlayAuditUploadedCount = 0;
        mobileOverlayAuditServerMaximumFrames = 0;
        mobileOverlayAuditNextCaptureSeconds = -1.0f;
        mobileOverlayAuditPoseTargets = null;
        mobileOverlayAuditTargetCaptured = null;
        mobileOverlayAuditServerTranslationToleranceMeters =
            mobileOverlayTargetTranslationToleranceMeters;
        mobileOverlayAuditServerRotationToleranceDegrees =
            mobileOverlayTargetRotationToleranceDegrees;
        mobileOverlayAuditNextGuidanceUpdateSeconds = -1.0f;
        if (poseDiagnosticRecordButton != null && reconstructedMeshRoot != null)
        {
            poseDiagnosticRecordButton.gameObject.SetActive(true);
            poseDiagnosticRecordButton.interactable =
                enableMobileOverlayAudit && !mobileOverlayAuditStartPending;
            SetButtonLabel(
                poseDiagnosticRecordButton,
                mobileOverlayAuditStartPending
                    ? "正在准备诊断..."
                    : "录制位姿诊断"
            );
            RefreshMeshControlDockVisibility();
        }
    }

    MobileOverlayAuditRequest BuildMobileOverlayAuditRequest()
    {
        if (
            activeMobileARResponse == null
            || string.IsNullOrEmpty(activeServerSessionId)
            || string.IsNullOrEmpty(preparedRuntimeOSha256)
            || string.IsNullOrEmpty(preparedRequestedPoseSha256)
        )
            throw new System.InvalidOperationException(
                "当前 Mesh 缺少手机渲染审计所需的不可变绑定"
            );
        return new MobileOverlayAuditRequest
        {
            session_id = activeServerSessionId,
            lifecycle_generation = arSessionLifecycleGeneration,
            runtime_o_sha256 = preparedRuntimeOSha256,
            requested_pose_sha256 = preparedRequestedPoseSha256,
            maximum_frames = 8,
            strict_reconstruction_input_pose_matching = true,
            target_translation_tolerance_meters = Mathf.Clamp(
                mobileOverlayTargetTranslationToleranceMeters,
                0.005f,
                0.10f
            ),
            target_rotation_tolerance_degrees = Mathf.Clamp(
                mobileOverlayTargetRotationToleranceDegrees,
                0.5f,
                15.0f
            ),
            diagnostic_stage = mobileOverlayAuditDiagnosticStage,
            alignment_refinement_state = alignmentRefinementState.ToString(),
            last_alignment_refinement_accepted = lastAlignmentRefinementAccepted,
            last_alignment_refinement_report = lastAlignmentRefinementReport,
            current_mesh_transform_unity = CurrentReconstructedMeshTransform(),
        };
    }

    Vector3 MobileOverlayTargetPosition(MobileOverlayPoseTarget target)
    {
        return new Vector3(target.position_x, target.position_y, target.position_z);
    }

    Quaternion MobileOverlayTargetRotation(MobileOverlayPoseTarget target)
    {
        Quaternion rotation = new Quaternion(
            target.quaternion_x,
            target.quaternion_y,
            target.quaternion_z,
            target.quaternion_w
        );
        float norm = Mathf.Sqrt(
            rotation.x * rotation.x
            + rotation.y * rotation.y
            + rotation.z * rotation.z
            + rotation.w * rotation.w
        );
        if (norm <= 1.0e-6f) return Quaternion.identity;
        return new Quaternion(
            rotation.x / norm,
            rotation.y / norm,
            rotation.z / norm,
            rotation.w / norm
        );
    }

    bool TryFindNearestUncapturedMobileOverlayTarget(
        Vector3 cameraPosition,
        Quaternion cameraRotation,
        out int targetArrayIndex,
        out float translationMeters,
        out float rotationDegrees
    )
    {
        targetArrayIndex = -1;
        translationMeters = float.PositiveInfinity;
        rotationDegrees = float.PositiveInfinity;
        if (
            mobileOverlayAuditPoseTargets == null
            || mobileOverlayAuditTargetCaptured == null
            || mobileOverlayAuditPoseTargets.Length
                != mobileOverlayAuditTargetCaptured.Length
        )
            return false;
        float translationScale = Mathf.Max(
            1.0e-6f,
            mobileOverlayAuditServerTranslationToleranceMeters
        );
        float rotationScale = Mathf.Max(
            1.0e-6f,
            mobileOverlayAuditServerRotationToleranceDegrees
        );
        float bestScore = float.PositiveInfinity;
        for (int i = 0; i < mobileOverlayAuditPoseTargets.Length; i++)
        {
            if (mobileOverlayAuditTargetCaptured[i]) continue;
            MobileOverlayPoseTarget target = mobileOverlayAuditPoseTargets[i];
            if (target == null) continue;
            float translation = Vector3.Distance(
                cameraPosition,
                MobileOverlayTargetPosition(target)
            );
            float rotation = Quaternion.Angle(
                cameraRotation,
                MobileOverlayTargetRotation(target)
            );
            float score =
                translation * translation
                    / (translationScale * translationScale)
                + rotation * rotation / (rotationScale * rotationScale);
            if (score >= bestScore) continue;
            bestScore = score;
            targetArrayIndex = i;
            translationMeters = translation;
            rotationDegrees = rotation;
        }
        return targetArrayIndex >= 0;
    }

    bool MobileOverlayPosePassesTarget(
        Vector3 cameraPosition,
        Quaternion cameraRotation,
        MobileOverlayPoseTarget target,
        out float translationMeters,
        out float rotationDegrees
    )
    {
        translationMeters = Vector3.Distance(
            cameraPosition,
            MobileOverlayTargetPosition(target)
        );
        rotationDegrees = Quaternion.Angle(
            cameraRotation,
            MobileOverlayTargetRotation(target)
        );
        return translationMeters
                <= mobileOverlayAuditServerTranslationToleranceMeters
            && rotationDegrees
                <= mobileOverlayAuditServerRotationToleranceDegrees;
    }

    void UpdateMobileOverlayAudit()
    {
        if (
            !enableMobileOverlayAudit
            || reconstructedMeshRoot == null
            || activeMobileARResponse == null
            || applicationPaused
            || ARSession.state != ARSessionState.SessionTracking
            || !CaptureReferencePoseStable()
        )
            return;

        if (mobileOverlayAuditStartPending)
        {
            if (!mobileOverlayAuditStartInFlight)
                StartCoroutine(StartMobileOverlayAudit());
            return;
        }
        if (
            !mobileOverlayAuditCaptureActive
            || mobileOverlayAuditSending
            || reconstructedMeshDisplayMode != 0
            || !cameraFrameHasCaptureAnchorPose
            || alignmentRefinementState == AlignmentRefinementState.Capturing
            || alignmentRefinementState == AlignmentRefinementState.Optimizing
        )
            return;

        int maximum = Mathf.Max(1, mobileOverlayAuditServerMaximumFrames);
        if (mobileOverlayAuditUploadedCount >= maximum)
        {
            mobileOverlayAuditCaptureActive = false;
            if (poseDiagnosticRecordButton != null)
            {
                poseDiagnosticRecordButton.interactable = true;
                SetButtonLabel(poseDiagnosticRecordButton, "再次录制诊断");
            }
            return;
        }
        float now = Time.realtimeSinceStartup;
        if (
            mobileOverlayAuditNextCaptureSeconds >= 0.0f
            && now < mobileOverlayAuditNextCaptureSeconds
        )
            return;
        if (
            !TryFindNearestUncapturedMobileOverlayTarget(
                cameraFrameCaptureAnchorPosition,
                cameraFrameCaptureAnchorRotation,
                out int targetArrayIndex,
                out float translationMeters,
                out float rotationDegrees
            )
        )
            return;
        MobileOverlayPoseTarget target = mobileOverlayAuditPoseTargets[
            targetArrayIndex
        ];
        if (now >= mobileOverlayAuditNextGuidanceUpdateSeconds)
        {
            mobileOverlayAuditNextGuidanceUpdateSeconds = now + 0.25f;
            UpdateUI(
                $"严格匹配重建输入 {mobileOverlayAuditUploadedCount + 1}/"
                    + $"{maximum}：请对准 {target.source_frame_name}\n"
                    + $"位置差 {translationMeters * 100.0f:F1} cm / "
                    + $"{mobileOverlayAuditServerTranslationToleranceMeters * 100.0f:F1} cm，"
                    + $"角度差 {rotationDegrees:F1}° / "
                    + $"{mobileOverlayAuditServerRotationToleranceDegrees:F1}°",
                Color.yellow
            );
        }
        if (
            translationMeters
                > mobileOverlayAuditServerTranslationToleranceMeters
            || rotationDegrees
                > mobileOverlayAuditServerRotationToleranceDegrees
        )
            return;
        StartCoroutine(
            CaptureAndUploadMobileOverlayAuditFrame(targetArrayIndex)
        );
    }

    IEnumerator StartMobileOverlayAudit()
    {
        int generation = mobileOverlayAuditGeneration;
        mobileOverlayAuditStartInFlight = true;
        MobileOverlayAuditRequest requestData;
        try
        {
            requestData = BuildMobileOverlayAuditRequest();
        }
        catch (System.Exception exception)
        {
            mobileOverlayAuditStartPending = false;
            mobileOverlayAuditStartInFlight = false;
            Debug.LogWarning(
                "[MobileOverlayAudit] start contract unavailable: "
                + exception.Message
            );
            if (poseDiagnosticRecordButton != null)
            {
                poseDiagnosticRecordButton.interactable = true;
                SetButtonLabel(poseDiagnosticRecordButton, "重试位姿诊断");
            }
            yield break;
        }
        using (
            UnityWebRequest web = new UnityWebRequest(
                serverURL + "/mobile_overlay_audit/start",
                "POST"
            )
        )
        {
            byte[] body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(requestData));
            web.uploadHandler = new UploadHandlerRaw(body);
            web.downloadHandler = new DownloadHandlerBuffer();
            web.SetRequestHeader("Content-Type", "application/json");
            web.timeout = 60;
            yield return web.SendWebRequest();
            if (generation != mobileOverlayAuditGeneration) yield break;
            mobileOverlayAuditStartInFlight = false;
            mobileOverlayAuditStartPending = false;
            if (web.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning(
                    "[MobileOverlayAudit] start failed: "
                    + ExtractServerMessage(web)
                );
                if (poseDiagnosticRecordButton != null)
                {
                    poseDiagnosticRecordButton.interactable = true;
                    SetButtonLabel(poseDiagnosticRecordButton, "重试位姿诊断");
                }
                yield break;
            }
            MobileOverlayAuditStartResponse response = null;
            try
            {
                response = JsonUtility.FromJson<MobileOverlayAuditStartResponse>(
                    web.downloadHandler.text
                );
            }
            catch
            {
                response = null;
            }
            bool poseTargetContractValid =
                response != null
                && response.strict_reconstruction_input_pose_matching
                && response.maximum_frames == 8
                && response.pose_targets != null
                && response.pose_targets.Length == response.maximum_frames
                && response.target_translation_tolerance_meters >= 0.005f
                && response.target_translation_tolerance_meters <= 0.10f
                && response.target_rotation_tolerance_degrees >= 0.5f
                && response.target_rotation_tolerance_degrees <= 15.0f;
            MobileOverlayPoseTarget[] orderedTargets = null;
            if (poseTargetContractValid)
            {
                orderedTargets = new MobileOverlayPoseTarget[
                    response.maximum_frames
                ];
                for (int i = 0; i < response.pose_targets.Length; i++)
                {
                    MobileOverlayPoseTarget target = response.pose_targets[i];
                    if (
                        target == null
                        || target.target_index < 0
                        || target.target_index >= orderedTargets.Length
                        || orderedTargets[target.target_index] != null
                        || string.IsNullOrEmpty(target.source_frame_name)
                    )
                    {
                        poseTargetContractValid = false;
                        break;
                    }
                    orderedTargets[target.target_index] = target;
                }
            }
            if (
                response == null
                || response.status != "capturing"
                || response.session_id != activeServerSessionId
                || response.lifecycle_generation != arSessionLifecycleGeneration
                || string.IsNullOrEmpty(response.audit_id)
                || !response.diagnostic_only
                || !poseTargetContractValid
            )
            {
                Debug.LogWarning(
                    "[MobileOverlayAudit] server returned an invalid start contract"
                );
                if (poseDiagnosticRecordButton != null)
                {
                    poseDiagnosticRecordButton.interactable = true;
                    SetButtonLabel(poseDiagnosticRecordButton, "重试位姿诊断");
                }
                yield break;
            }
            activeMobileOverlayAuditId = response.audit_id;
            mobileOverlayAuditServerMaximumFrames = Mathf.Max(
                1,
                response.maximum_frames
            );
            mobileOverlayAuditPoseTargets = orderedTargets;
            mobileOverlayAuditTargetCaptured = new bool[
                mobileOverlayAuditPoseTargets.Length
            ];
            mobileOverlayAuditServerTranslationToleranceMeters =
                response.target_translation_tolerance_meters;
            mobileOverlayAuditServerRotationToleranceDegrees =
                response.target_rotation_tolerance_degrees;
            mobileOverlayAuditUploadedCount = 0;
            mobileOverlayAuditNextCaptureSeconds = Time.realtimeSinceStartup;
            mobileOverlayAuditNextGuidanceUpdateSeconds = -1.0f;
            mobileOverlayAuditCaptureActive = true;
            if (poseDiagnosticRecordButton != null)
            {
                poseDiagnosticRecordButton.interactable = false;
                SetButtonLabel(
                    poseDiagnosticRecordButton,
                    $"诊断录制 0/{mobileOverlayAuditServerMaximumFrames}"
                );
            }
            Debug.Log(
                "[MobileOverlayAudit] low-frequency diagnostic started: "
                + response.report
            );
        }
    }

    Texture2D DownscaleAuditTexture(Texture2D source, int maximumLongEdge)
    {
        if (source == null) return null;
        int limit = Mathf.Max(64, maximumLongEdge);
        int longest = Mathf.Max(source.width, source.height);
        if (longest <= limit) return source;
        float ratio = (float)limit / longest;
        int width = Mathf.Max(1, Mathf.RoundToInt(source.width * ratio));
        int height = Mathf.Max(1, Mathf.RoundToInt(source.height * ratio));
        RenderTexture temporary = RenderTexture.GetTemporary(
            width,
            height,
            0,
            RenderTextureFormat.ARGB32
        );
        RenderTexture previous = RenderTexture.active;
        Graphics.Blit(source, temporary);
        RenderTexture.active = temporary;
        Texture2D resized = new Texture2D(
            width,
            height,
            TextureFormat.RGBA32,
            false
        );
        resized.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
        resized.Apply(false, false);
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(temporary);
        return resized;
    }

    Texture2D ReadAuditRenderTexture(RenderTexture source)
    {
        if (source == null || !source.IsCreated()) return null;
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = source;
        Texture2D result = new Texture2D(
            source.width,
            source.height,
            TextureFormat.RGBA32,
            false
        );
        result.ReadPixels(
            new Rect(0, 0, source.width, source.height),
            0,
            0,
            false
        );
        result.Apply(false, false);
        RenderTexture.active = previous;
        return result;
    }

    bool TryEncodeLatestDiagnosticCamera(
        out byte[] jpegBytes,
        out int imageWidth,
        out int imageHeight,
        out double cpuImageTimestampSeconds
    )
    {
        jpegBytes = null;
        imageWidth = 0;
        imageHeight = 0;
        cpuImageTimestampSeconds = -1.0;
        if (
            cameraManager == null
            || !cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image)
        )
            return false;

        NativeArray<byte> buffer = default(NativeArray<byte>);
        Texture2D texture = null;
        try
        {
            cpuImageTimestampSeconds = image.timestamp;
            imageWidth = image.width;
            imageHeight = image.height;
            XRCpuImage.ConversionParams conversion =
                new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(0, 0, image.width, image.height),
                    outputDimensions = new Vector2Int(image.width, image.height),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.None,
                };
            int size = image.GetConvertedDataSize(conversion);
            buffer = new NativeArray<byte>(size, Allocator.Temp);
            image.Convert(conversion, buffer);
            texture = new Texture2D(
                image.width,
                image.height,
                TextureFormat.RGBA32,
                false
            );
            texture.LoadRawTextureData(buffer);
            texture.Apply(false, false);
            jpegBytes = texture.EncodeToJPG(
                Mathf.Clamp(mobileOverlayAuditJpegQuality, 80, 100)
            );
            return jpegBytes != null && jpegBytes.Length > 0;
        }
        catch (System.Exception exception)
        {
            Debug.LogWarning(
                "[MobileOverlayAudit] raw camera capture failed: "
                    + exception.Message
            );
            jpegBytes = null;
            return false;
        }
        finally
        {
            if (buffer.IsCreated) buffer.Dispose();
            image.Dispose();
            if (texture != null) Destroy(texture);
        }
    }

    IEnumerator CaptureAndUploadMobileOverlayAuditFrame(int targetArrayIndex)
    {
        if (mobileOverlayAuditSending) yield break;
        if (
            mobileOverlayAuditPoseTargets == null
            || mobileOverlayAuditTargetCaptured == null
            || targetArrayIndex < 0
            || targetArrayIndex >= mobileOverlayAuditPoseTargets.Length
            || mobileOverlayAuditTargetCaptured[targetArrayIndex]
        )
            yield break;
        MobileOverlayPoseTarget target = mobileOverlayAuditPoseTargets[
            targetArrayIndex
        ];
        int generation = mobileOverlayAuditGeneration;
        string auditId = activeMobileOverlayAuditId;
        mobileOverlayAuditSending = true;

        if (
            !cameraFrameHasCaptureAnchorPose
            || !cameraFrameHasIntrinsics
            || !hasCameraFrameSnapshot
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        Vector3 rawCameraA0Position = cameraFrameCaptureAnchorPosition;
        Quaternion rawCameraA0Rotation = cameraFrameCaptureAnchorRotation;
        if (
            !MobileOverlayPosePassesTarget(
                rawCameraA0Position,
                rawCameraA0Rotation,
                target,
                out float rawTargetTranslationMeters,
                out float rawTargetRotationDegrees
            )
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        XRCameraIntrinsics rawIntrinsics = cameraFrameIntrinsics;
        long rawCameraFrameTimestampNs = cameraFrameTimestampNs;
        double rawPoseSampleSeconds = cameraFramePoseSampleSeconds;
        Matrix4x4 rawDisplayMatrix = cameraFrameDisplayMatrix;
        Matrix4x4 rawProjectionMatrix = cameraFrameProjectionMatrix;
        if (
            !TryEncodeLatestDiagnosticCamera(
                out byte[] rawCameraBytes,
                out int rawCpuImageWidth,
                out int rawCpuImageHeight,
                out double rawCpuTimestampSeconds
            )
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        double rawFrameTimestampSeconds = rawCameraFrameTimestampNs > 0
            ? rawCameraFrameTimestampNs * 1.0e-9
            : -1.0;
        double rawTimestampDeltaSeconds =
            rawFrameTimestampSeconds > 0.0 && rawCpuTimestampSeconds > 0.0
                ? System.Math.Abs(
                    rawCpuTimestampSeconds - rawFrameTimestampSeconds
                )
                : -1.0;
        if (
            rawTimestampDeltaSeconds < 0.0
            || rawTimestampDeltaSeconds > 0.10
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }

        yield return new WaitForEndOfFrame();
        if (
            generation != mobileOverlayAuditGeneration
            || !mobileOverlayAuditCaptureActive
            || reconstructedMeshRoot == null
            || arCamera == null
            || !CaptureReferencePoseStable()
            || string.IsNullOrEmpty(auditId)
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }

        if (
            !TryGetPoseRelativeToCaptureAnchor(
                arCamera.position,
                arCamera.rotation,
                out Vector3 cameraA0Position,
                out Quaternion cameraA0Rotation
            )
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        if (
            !MobileOverlayPosePassesTarget(
                cameraA0Position,
                cameraA0Rotation,
                target,
                out float screenTargetTranslationMeters,
                out float screenTargetRotationDegrees
            )
        )
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        float poseSampleSeconds = Time.realtimeSinceStartup;
        Texture2D fullScreen = ScreenCapture.CaptureScreenshotAsTexture();
        float screenCaptureSeconds = Time.realtimeSinceStartup;
        if (fullScreen == null)
        {
            mobileOverlayAuditSending = false;
            yield break;
        }
        if (screenCaptureSeconds - poseSampleSeconds > 0.10f)
        {
            Destroy(fullScreen);
            mobileOverlayAuditSending = false;
            yield break;
        }
        Texture2D composite = mobileOverlayAuditKeepNativeScreenResolution
            ? fullScreen
            : DownscaleAuditTexture(fullScreen, mobileOverlayAuditMaxLongEdge);
        if (composite != fullScreen) Destroy(fullScreen);
        byte[] compositeBytes = composite.EncodeToPNG();
        Destroy(composite);

        byte[] outlineBytes = null;
        if (
            reconstructedOutlineMethod
                == ReconstructedOutlineMethod.ServerStyleScreenSpace
            && reconstructedServerMaskTexture != null
        )
        {
            Texture2D outlineFull = ReadAuditRenderTexture(
                reconstructedServerMaskTexture
            );
            if (outlineFull != null)
            {
                Texture2D outline = DownscaleAuditTexture(
                    outlineFull,
                    mobileOverlayAuditKeepNativeScreenResolution
                        ? Mathf.Max(outlineFull.width, outlineFull.height)
                        : mobileOverlayAuditMaxLongEdge
                );
                if (outline != outlineFull) Destroy(outlineFull);
                outlineBytes = outline.EncodeToPNG();
                Destroy(outline);
            }
        }

        MeshTransformUnity meshTransform = CurrentReconstructedMeshTransform();
        Transform a0Transform = captureReferenceAnchorObject.transform;
        Vector3 a0WorldPosition = a0Transform.position;
        Quaternion a0WorldRotation = a0Transform.rotation;
        Vector3 cameraWorldPosition = arCamera.position;
        Quaternion cameraWorldRotation = arCamera.rotation;
        Transform meshWorldTransform = reconstructedMeshRoot.transform;
        Vector3 meshWorldPosition = meshWorldTransform.position;
        Quaternion meshWorldRotation = meshWorldTransform.rotation;
        Vector3 meshWorldScale = meshWorldTransform.lossyScale;
        string anchorTrackableId = captureReferenceAnchor != null
            ? captureReferenceAnchor.trackableId.ToString()
            : "";
        WWWForm form = new WWWForm();
        form.AddField("session_id", activeServerSessionId);
        form.AddField(
            "lifecycle_generation",
            IntString(arSessionLifecycleGeneration)
        );
        form.AddField("audit_id", auditId);
        form.AddField("overlay_contract", MobileOverlayAuditContract);
        form.AddField("target_index", IntString(target.target_index));
        form.AddField(
            "target_source_frame_name",
            target.source_frame_name ?? ""
        );
        form.AddField(
            "client_raw_target_translation_meters",
            FloatString(rawTargetTranslationMeters)
        );
        form.AddField(
            "client_raw_target_rotation_degrees",
            FloatString(rawTargetRotationDegrees)
        );
        form.AddField(
            "client_screen_target_translation_meters",
            FloatString(screenTargetTranslationMeters)
        );
        form.AddField(
            "client_screen_target_rotation_degrees",
            FloatString(screenTargetRotationDegrees)
        );
        form.AddField("screen_capture_encoding", "png_lossless_native");
        form.AddField("capture_anchor_tracking_state", "Tracking");
        form.AddField("camera_pos_x", FloatString(cameraA0Position.x));
        form.AddField("camera_pos_y", FloatString(cameraA0Position.y));
        form.AddField("camera_pos_z", FloatString(cameraA0Position.z));
        form.AddField("camera_quat_x", FloatString(cameraA0Rotation.x));
        form.AddField("camera_quat_y", FloatString(cameraA0Rotation.y));
        form.AddField("camera_quat_z", FloatString(cameraA0Rotation.z));
        form.AddField("camera_quat_w", FloatString(cameraA0Rotation.w));
        form.AddField("raw_camera_pos_x", FloatString(rawCameraA0Position.x));
        form.AddField("raw_camera_pos_y", FloatString(rawCameraA0Position.y));
        form.AddField("raw_camera_pos_z", FloatString(rawCameraA0Position.z));
        form.AddField("raw_camera_quat_x", FloatString(rawCameraA0Rotation.x));
        form.AddField("raw_camera_quat_y", FloatString(rawCameraA0Rotation.y));
        form.AddField("raw_camera_quat_z", FloatString(rawCameraA0Rotation.z));
        form.AddField("raw_camera_quat_w", FloatString(rawCameraA0Rotation.w));
        form.AddField("mesh_pos_x", FloatString(meshTransform.position_x));
        form.AddField("mesh_pos_y", FloatString(meshTransform.position_y));
        form.AddField("mesh_pos_z", FloatString(meshTransform.position_z));
        form.AddField("mesh_quat_x", FloatString(meshTransform.quaternion_x));
        form.AddField("mesh_quat_y", FloatString(meshTransform.quaternion_y));
        form.AddField("mesh_quat_z", FloatString(meshTransform.quaternion_z));
        form.AddField("mesh_quat_w", FloatString(meshTransform.quaternion_w));
        form.AddField(
            "mesh_uniform_scale",
            FloatString(meshTransform.uniform_scale)
        );
        form.AddField(
            "screen_capture_realtime_s",
            FloatString(screenCaptureSeconds)
        );
        form.AddField(
            "camera_pose_sample_realtime_s",
            FloatString(poseSampleSeconds)
        );
        form.AddField(
            "camera_pose_to_screen_capture_delta_s",
            FloatString(screenCaptureSeconds - poseSampleSeconds)
        );
        form.AddField(
            "raw_cpu_image_timestamp_s",
            DoubleString(rawCpuTimestampSeconds)
        );
        form.AddField(
            "raw_camera_frame_timestamp_ns",
            rawCameraFrameTimestampNs.ToString(CultureInfo.InvariantCulture)
        );
        form.AddField(
            "raw_pose_sample_realtime_s",
            DoubleString(rawPoseSampleSeconds)
        );
        form.AddField(
            "raw_cpu_to_camera_frame_timestamp_delta_s",
            DoubleString(rawTimestampDeltaSeconds)
        );
        form.AddField("raw_cpu_image_width", IntString(rawCpuImageWidth));
        form.AddField("raw_cpu_image_height", IntString(rawCpuImageHeight));
        form.AddField("raw_image_transform", CpuImageTransformName);
        form.AddField("fx", FloatString(rawIntrinsics.focalLength.x));
        form.AddField("fy", FloatString(rawIntrinsics.focalLength.y));
        form.AddField("cx", FloatString(rawIntrinsics.principalPoint.x));
        form.AddField("cy", FloatString(rawIntrinsics.principalPoint.y));
        form.AddField("intrinsic_width", IntString(rawIntrinsics.resolution.x));
        form.AddField("intrinsic_height", IntString(rawIntrinsics.resolution.y));
        form.AddField("screen_width", IntString(Screen.width));
        form.AddField("screen_height", IntString(Screen.height));
        form.AddField("screen_orientation", Screen.orientation.ToString());
        form.AddField("outline_method", reconstructedOutlineMethod.ToString());
        form.AddField(
            "outline_display_requested",
            reconstructedMeshDisplayMode == 0 ? "true" : "false"
        );
        form.AddField("display_matrix", MatrixString(cameraFrameDisplayMatrix));
        form.AddField(
            "projection_matrix",
            MatrixString(cameraFrameProjectionMatrix)
        );
        form.AddField("raw_display_matrix", MatrixString(rawDisplayMatrix));
        form.AddField("raw_projection_matrix", MatrixString(rawProjectionMatrix));
        form.AddField("a0_world_pos_x", FloatString(a0WorldPosition.x));
        form.AddField("a0_world_pos_y", FloatString(a0WorldPosition.y));
        form.AddField("a0_world_pos_z", FloatString(a0WorldPosition.z));
        form.AddField("a0_world_quat_x", FloatString(a0WorldRotation.x));
        form.AddField("a0_world_quat_y", FloatString(a0WorldRotation.y));
        form.AddField("a0_world_quat_z", FloatString(a0WorldRotation.z));
        form.AddField("a0_world_quat_w", FloatString(a0WorldRotation.w));
        form.AddField("camera_world_pos_x", FloatString(cameraWorldPosition.x));
        form.AddField("camera_world_pos_y", FloatString(cameraWorldPosition.y));
        form.AddField("camera_world_pos_z", FloatString(cameraWorldPosition.z));
        form.AddField("camera_world_quat_x", FloatString(cameraWorldRotation.x));
        form.AddField("camera_world_quat_y", FloatString(cameraWorldRotation.y));
        form.AddField("camera_world_quat_z", FloatString(cameraWorldRotation.z));
        form.AddField("camera_world_quat_w", FloatString(cameraWorldRotation.w));
        form.AddField("mesh_world_pos_x", FloatString(meshWorldPosition.x));
        form.AddField("mesh_world_pos_y", FloatString(meshWorldPosition.y));
        form.AddField("mesh_world_pos_z", FloatString(meshWorldPosition.z));
        form.AddField("mesh_world_quat_x", FloatString(meshWorldRotation.x));
        form.AddField("mesh_world_quat_y", FloatString(meshWorldRotation.y));
        form.AddField("mesh_world_quat_z", FloatString(meshWorldRotation.z));
        form.AddField("mesh_world_quat_w", FloatString(meshWorldRotation.w));
        form.AddField("mesh_world_scale_x", FloatString(meshWorldScale.x));
        form.AddField("mesh_world_scale_y", FloatString(meshWorldScale.y));
        form.AddField("mesh_world_scale_z", FloatString(meshWorldScale.z));
        form.AddField("capture_anchor_trackable_id", anchorTrackableId);
        form.AddField(
            "capture_anchor_pose_valid",
            captureReferenceAnchorPoseValid ? "true" : "false"
        );
        form.AddField(
            "capture_anchor_tracking_stable",
            captureReferenceAnchorTrackingStable ? "true" : "false"
        );
        form.AddField(
            "capture_anchor_ever_tracked",
            captureReferenceAnchorEverTracked ? "true" : "false"
        );
        form.AddField(
            "capture_anchor_uses_tracked_ar_anchor",
            captureReferenceUsesTrackedARAnchor ? "true" : "false"
        );
        form.AddField(
            "capture_anchor_tracking_since_realtime_s",
            FloatString(captureReferenceAnchorTrackingSince)
        );
        form.AddField("ar_session_state", ARSession.state.ToString());
        form.AddField("application_paused", applicationPaused ? "true" : "false");
        form.AddField("application_focused", Application.isFocused ? "true" : "false");
        form.AddField("camera_frame_sequence", IntString(cameraFrameSequence));
        form.AddField("device_model", SystemInfo.deviceModel ?? "");
        form.AddField("operating_system", SystemInfo.operatingSystem ?? "");
        form.AddField("application_version", Application.version ?? "");
        form.AddField("battery_level", FloatString(SystemInfo.batteryLevel));
        form.AddField("battery_status", SystemInfo.batteryStatus.ToString());
        form.AddField(
            "alignment_refinement_state",
            alignmentRefinementState.ToString()
        );
        form.AddField("diagnostic_stage", mobileOverlayAuditDiagnosticStage);
        form.AddField("mobile_realtime_s", FloatString(Time.realtimeSinceStartup));
        form.AddBinaryData(
            "composite",
            compositeBytes,
            "mobile_screen_composite.png",
            "image/png"
        );
        form.AddBinaryData(
            "raw_camera",
            rawCameraBytes,
            "raw_xrcpuimage.jpg",
            "image/jpeg"
        );
        if (outlineBytes != null && outlineBytes.Length > 0)
            form.AddBinaryData(
                "outline",
                outlineBytes,
                "mobile_outline.png",
                "image/png"
            );

        using (
            UnityWebRequest web = UnityWebRequest.Post(
                serverURL + "/mobile_overlay_audit/upload",
                form
            )
        )
        {
            web.timeout = 60;
            yield return web.SendWebRequest();
            if (generation != mobileOverlayAuditGeneration) yield break;
            if (web.result == UnityWebRequest.Result.Success)
            {
                MobileOverlayAuditUploadResponse response = null;
                try
                {
                    response = JsonUtility.FromJson<MobileOverlayAuditUploadResponse>(
                        web.downloadHandler.text
                    );
                }
                catch
                {
                    response = null;
                }
                if (
                    response != null
                    && response.status == "success"
                    && response.session_id == activeServerSessionId
                    && response.audit_id == auditId
                    && response.diagnostic_only
                    && response.matched_target_index == target.target_index
                    && response.matched_target_source_frame_name
                        == target.source_frame_name
                )
                {
                    mobileOverlayAuditTargetCaptured[targetArrayIndex] = true;
                    mobileOverlayAuditUploadedCount = response.captured_frames;
                    mobileOverlayAuditNextCaptureSeconds =
                        Time.realtimeSinceStartup
                        + Mathf.Max(0.5f, mobileOverlayAuditIntervalSeconds);
                    if (poseDiagnosticRecordButton != null)
                    {
                        SetButtonLabel(
                            poseDiagnosticRecordButton,
                            $"严格同位姿 {response.captured_frames}/"
                                + $"{response.maximum_frames}"
                        );
                    }
                    if (response.complete)
                    {
                        mobileOverlayAuditCaptureActive = false;
                        if (poseDiagnosticRecordButton != null)
                        {
                            poseDiagnosticRecordButton.interactable = true;
                            SetButtonLabel(
                                poseDiagnosticRecordButton,
                                "再次录制诊断"
                            );
                        }
                        UpdateUI(
                            "严格匹配重建8帧的诊断录制完成；服务器已生成方向统一的高清对照",
                            Color.green
                        );
                        Debug.Log(
                            "[MobileOverlayAudit] complete: " + response.report
                        );
                    }
                }
                else
                {
                    Debug.LogWarning(
                        "[MobileOverlayAudit] invalid upload response contract"
                    );
                }
            }
            else
            {
                mobileOverlayAuditNextCaptureSeconds =
                    Time.realtimeSinceStartup
                    + Mathf.Max(0.5f, mobileOverlayAuditIntervalSeconds);
                Debug.LogWarning(
                    "[MobileOverlayAudit] upload failed: "
                    + ExtractServerMessage(web)
                );
            }
        }
        mobileOverlayAuditSending = false;
    }

    AlignmentRefineRequest BuildAlignmentRefineRequest()
    {
        if (activeMobileARResponse == null)
            throw new System.InvalidOperationException(
                "当前 Mesh 缺少不可变服务端绑定"
            );
        return new AlignmentRefineRequest
        {
            session_id = activeServerSessionId,
            lifecycle_generation = arSessionLifecycleGeneration,
            runtime_o_sha256 = preparedRuntimeOSha256,
            requested_pose_sha256 = preparedRequestedPoseSha256,
            refinement_id = activeAlignmentRefinementId,
            current_mesh_transform_unity = CurrentReconstructedMeshTransform(),
        };
    }

    public void ToggleFastAlignmentRefinement()
    {
        if (
            mobileOverlayAuditStartPending
            || mobileOverlayAuditStartInFlight
            || mobileOverlayAuditCaptureActive
            || mobileOverlayAuditSending
        )
        {
            UpdateUI(
                "请先完成当前位姿诊断录制，再开始快速校准",
                Color.yellow
            );
            return;
        }
        if (
            alignmentRefinementState == AlignmentRefinementState.Optimizing
            || isSending
        )
            return;
        if (alignmentRefinementState == AlignmentRefinementState.Capturing)
        {
            if (alignmentRefinementUploadedCount < Mathf.Max(16, alignmentRefineMinimumFrames))
            {
                UpdateUI(
                    $"快速校准至少需要 {Mathf.Max(16, alignmentRefineMinimumFrames)} 个候选帧；"
                    + $"当前 {alignmentRefinementUploadedCount} 帧，请继续缓慢移动",
                    Color.yellow
                );
                return;
            }
            StartCoroutine(OptimizeFastAlignmentRefinement());
            return;
        }
        if (
            alignmentRefinementState == AlignmentRefinementState.Ready
            || alignmentRefinementState == AlignmentRefinementState.Complete
        )
            StartCoroutine(StartFastAlignmentRefinement());
    }

    IEnumerator StartFastAlignmentRefinement()
    {
        if (
            reconstructedMeshRoot == null
            || activeMobileARResponse == null
            || !CaptureReferencePoseStable()
        )
        {
            UpdateUI(
                "快速校准需要已显示的 Mesh 和稳定 Tracking 的采集 A0",
                Color.yellow
            );
            yield break;
        }
        AlignmentRefineRequest requestData;
        try
        {
            requestData = BuildAlignmentRefineRequest();
        }
        catch (System.Exception exception)
        {
            UpdateUI("无法建立快速校准合同: " + exception.Message, Color.red);
            yield break;
        }
        alignmentRefineButton.interactable = false;
        SetButtonLabel(alignmentRefineButton, "正在准备校准...");
        using (
            UnityWebRequest web = new UnityWebRequest(
                serverURL + "/alignment_refine/start",
                "POST"
            )
        )
        {
            byte[] body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(requestData));
            web.uploadHandler = new UploadHandlerRaw(body);
            web.downloadHandler = new DownloadHandlerBuffer();
            web.SetRequestHeader("Content-Type", "application/json");
            web.timeout = 120;
            yield return web.SendWebRequest();
            if (web.result != UnityWebRequest.Result.Success)
            {
                alignmentRefinementState = AlignmentRefinementState.Ready;
                alignmentRefineButton.interactable = true;
                SetButtonLabel(alignmentRefineButton, "快速校准");
                UpdateUI("快速校准准备失败: " + ExtractServerMessage(web), Color.red);
                yield break;
            }
            AlignmentRefineStartResponse response = null;
            try
            {
                response = JsonUtility.FromJson<AlignmentRefineStartResponse>(
                    web.downloadHandler.text
                );
            }
            catch
            {
                response = null;
            }
            if (
                response == null
                || response.status != "capturing"
                || response.session_id != activeServerSessionId
                || response.lifecycle_generation != arSessionLifecycleGeneration
                || string.IsNullOrEmpty(response.refinement_id)
                || response.minimum_frames < 16
                || response.optimization_views != 16
            )
            {
                alignmentRefinementState = AlignmentRefinementState.Ready;
                alignmentRefineButton.interactable = true;
                SetButtonLabel(alignmentRefineButton, "快速校准");
                UpdateUI("快速校准服务端响应合同无效", Color.red);
                yield break;
            }
            alignmentRefineMinimumFrames = response.minimum_frames;
            alignmentRefineRecommendedFrames = Mathf.Max(
                response.minimum_frames,
                response.recommended_frames
            );
            activeAlignmentRefinementId = response.refinement_id;
            alignmentRefinementUploadedCount = 0;
            alignmentRefinementTimer = Mathf.Max(0.10f, alignmentRefineSendInterval);
            alignmentRefinementState = AlignmentRefinementState.Capturing;
            alignmentRefineButton.interactable = true;
            SetButtonLabel(alignmentRefineButton, "结束采集并优化");
            UpdateUI(response.message, Color.green);
        }
    }

    IEnumerator OptimizeFastAlignmentRefinement()
    {
        alignmentRefinementState = AlignmentRefinementState.Optimizing;
        alignmentRefineButton.interactable = false;
        SetButtonLabel(alignmentRefineButton, "SAM2校准中...");
        AlignmentRefineRequest requestData;
        try
        {
            requestData = BuildAlignmentRefineRequest();
        }
        catch (System.Exception exception)
        {
            alignmentRefinementState = AlignmentRefinementState.Ready;
            alignmentRefineButton.interactable = true;
            SetButtonLabel(alignmentRefineButton, "重新快速校准");
            UpdateUI(
                "无法建立快速校准合同，原 Mesh 位姿未改变: " + exception.Message,
                Color.red
            );
            yield break;
        }
        requestData.refinement_id = activeAlignmentRefinementId;
        using (
            UnityWebRequest web = new UnityWebRequest(
                serverURL + "/alignment_refine/optimize",
                "POST"
            )
        )
        {
            byte[] body = Encoding.UTF8.GetBytes(JsonUtility.ToJson(requestData));
            web.uploadHandler = new UploadHandlerRaw(body);
            web.downloadHandler = new DownloadHandlerBuffer();
            web.SetRequestHeader("Content-Type", "application/json");
            web.timeout = 300;
            yield return web.SendWebRequest();
            alignmentRefineButton.interactable = true;
            if (web.result != UnityWebRequest.Result.Success)
            {
                alignmentRefinementState = AlignmentRefinementState.Ready;
                SetButtonLabel(alignmentRefineButton, "重新快速校准");
                UpdateUI(
                    "快速校准失败，原 Mesh 位姿未改变: "
                        + ExtractServerMessage(web),
                    Color.red
                );
                yield break;
            }
            AlignmentRefineOptimizeResponse response = null;
            try
            {
                response = JsonUtility.FromJson<AlignmentRefineOptimizeResponse>(
                    web.downloadHandler.text
                );
            }
            catch
            {
                response = null;
            }
            if (
                response == null
                || response.status != "success"
                || response.session_id != activeServerSessionId
                || response.lifecycle_generation != arSessionLifecycleGeneration
                || response.refinement_id != activeAlignmentRefinementId
                || response.geometry_regenerated
                || response.selected_mesh_transform_unity == null
            )
            {
                alignmentRefinementState = AlignmentRefinementState.Ready;
                SetButtonLabel(alignmentRefineButton, "重新快速校准");
                UpdateUI("快速校准结果合同无效，原 Mesh 位姿未改变", Color.red);
                yield break;
            }
            if (response.accepted)
            {
                MeshTransformUnity selected = response.selected_mesh_transform_unity;
                Quaternion rotation = new Quaternion(
                    selected.quaternion_x,
                    selected.quaternion_y,
                    selected.quaternion_z,
                    selected.quaternion_w
                );
                float norm = Mathf.Sqrt(
                    rotation.x * rotation.x
                    + rotation.y * rotation.y
                    + rotation.z * rotation.z
                    + rotation.w * rotation.w
                );
                Vector3 selectedPosition = new Vector3(
                    selected.position_x,
                    selected.position_y,
                    selected.position_z
                );
                if (
                    reconstructedMeshRoot == null
                    || !IsFinite(selectedPosition)
                    || !IsFinite(rotation)
                    || norm <= 1.0e-6f
                    || float.IsNaN(selected.uniform_scale)
                    || float.IsInfinity(selected.uniform_scale)
                    || selected.uniform_scale <= 0.0f
                )
                {
                    alignmentRefinementState = AlignmentRefinementState.Ready;
                    SetButtonLabel(alignmentRefineButton, "重新快速校准");
                    UpdateUI("校准变换包含无效数值，原 Mesh 位姿未改变", Color.red);
                    yield break;
                }
                rotation = new Quaternion(
                    rotation.x / norm,
                    rotation.y / norm,
                    rotation.z / norm,
                    rotation.w / norm
                );
                reconstructedMeshRoot.transform.localPosition = selectedPosition;
                reconstructedMeshRoot.transform.localRotation = rotation;
                reconstructedMeshRoot.transform.localScale =
                    Vector3.one * selected.uniform_scale;
                UpdateReconstructedSilhouette(true);
                ApplyReconstructedMeshDisplayMode();
            }
            lastAlignmentRefinementAccepted = response.accepted;
            lastAlignmentRefinementReport = response.report ?? "";
            alignmentRefinementState = AlignmentRefinementState.Complete;
            // Keep every pre-calibration attempt immutable.  The next explicit
            // recording (or the opt-in automatic mode) is now labelled from
            // the completed accepted/rejected calibration state.
            ResetMobileOverlayAudit(autoStartMobileOverlayAudit);
            SetButtonLabel(alignmentRefineButton, "再次快速校准");
            UpdateUI(
                response.message
                    + $"（IoU {response.initial_iou_mean:F3} → "
                    + $"{response.optimized_iou_mean:F3}，"
                    + $"Δ={response.iou_gain_mean:+0.000;-0.000;0.000}）",
                response.accepted ? Color.green : Color.yellow
            );
        }
    }

    public void ToggleReconstructedMeshDisplay()
    {
        if (reconstructedMeshRoot == null) return;
        reconstructedMeshDisplayMode = (reconstructedMeshDisplayMode + 1) % 2;
        if (
            reconstructedMeshDisplayMode == 0
            && reconstructedOutlineMethod
                == ReconstructedOutlineMethod.ViewDependentMeshLines
        )
            UpdateReconstructedSilhouette(true);
        ApplyReconstructedMeshDisplayMode();
    }

    public void ToggleReconstructedOutlineMethod()
    {
        if (reconstructedMeshRoot == null) return;
        if (!reconstructedServerStyleAvailable)
        {
            UpdateUI(
                "服务器式轮廓不可用：请在 Inspector 绑定 Mask 与 Screen Outline 两个材质",
                Color.yellow
            );
            return;
        }
        if (
            reconstructedOutlineMethod
                == ReconstructedOutlineMethod.ViewDependentMeshLines
            && reconstructedServerMaskCamera == null
        )
        {
            reconstructedServerStyleAvailable =
                TryCreateServerStyleOutlineResources(
                    reconstructedOutlineVertices,
                    reconstructedOutlineTriangles
                );
            if (!reconstructedServerStyleAvailable)
            {
                if (meshOutlineMethodButton != null)
                    meshOutlineMethodButton.gameObject.SetActive(false);
                RefreshMeshControlDockVisibility();
                UpdateUI(
                    "服务器式轮廓资源创建失败，已保留实时3D边显示",
                    Color.yellow
                );
                return;
            }
        }
        reconstructedOutlineMethod = reconstructedOutlineMethod
            == ReconstructedOutlineMethod.ViewDependentMeshLines
                ? ReconstructedOutlineMethod.ServerStyleScreenSpace
                : ReconstructedOutlineMethod.ViewDependentMeshLines;
        if (
            reconstructedOutlineMethod
            == ReconstructedOutlineMethod.ViewDependentMeshLines
        )
            UpdateReconstructedSilhouette(true);
        ApplyReconstructedMeshDisplayMode();
    }

    void ApplyReconstructedMeshDisplayMode()
    {
        bool anchorReady = captureReferenceAnchorTrackingStable;
        bool displayRequested = reconstructedMeshDisplayMode == 0;
        bool useServerStyle =
            reconstructedOutlineMethod
                == ReconstructedOutlineMethod.ServerStyleScreenSpace
            && reconstructedServerStyleAvailable;
        bool showMeshLines = anchorReady && displayRequested && !useServerStyle;
        bool showServerStyle = anchorReady && displayRequested && useServerStyle;
        if (reconstructedOutlineObject != null)
            reconstructedOutlineObject.SetActive(showMeshLines);
        if (reconstructedServerMaskObject != null)
            reconstructedServerMaskObject.SetActive(showServerStyle);
        if (reconstructedServerMaskCamera != null && !showServerStyle)
            reconstructedServerMaskCamera.enabled = false;
        if (reconstructedServerMaskCameraObject != null)
            reconstructedServerMaskCameraObject.SetActive(showServerStyle);
        if (reconstructedServerOutlineCanvasObject != null)
            reconstructedServerOutlineCanvasObject.SetActive(showServerStyle);
        if (showServerStyle) nextServerStyleRenderSeconds = -1.0f;

        string stateLabel;
        if (!anchorReady)
            stateLabel = "等待采集 A0 Anchor Tracking";
        else if (!displayRequested)
            stateLabel = "轮廓已隐藏";
        else
            stateLabel = useServerStyle
                ? "显示：服务器式屏幕轮廓"
                : "显示：实时3D剪影边";
        if (meshDisplayModeText != null) meshDisplayModeText.text = stateLabel;
        SetButtonLabel(
            meshDisplayButton,
            displayRequested ? "隐藏轮廓" : "显示轮廓"
        );

        string methodLabel = useServerStyle
            ? "方式：服务器式屏幕轮廓"
            : "方式：实时3D剪影边";
        if (meshOutlineMethodText != null)
            meshOutlineMethodText.text = methodLabel;
        SetButtonLabel(
            meshOutlineMethodButton,
            useServerStyle ? "切换：3D轮廓" : "切换：屏幕轮廓"
        );
        RefreshMeshControlDockVisibility();
    }

    void ClearReconstructedMesh()
    {
        ResetMobileOverlayAudit(false);
        activeMobileARResponse = null;
        alignmentRefinementState = AlignmentRefinementState.Unavailable;
        activeAlignmentRefinementId = "";
        alignmentRefinementUploadedCount = 0;
        alignmentRefinementTimer = 0.0f;
        lastAlignmentRefinementAccepted = false;
        lastAlignmentRefinementReport = "";
        ClearServerStyleOutlineResources();
        if (reconstructedMeshRoot != null) Destroy(reconstructedMeshRoot);
        if (reconstructedOutlineMesh != null) Destroy(reconstructedOutlineMesh);
        if (reconstructedOutlineMaterial != null) Destroy(reconstructedOutlineMaterial);
        reconstructedMeshRoot = null;
        reconstructedOutlineObject = null;
        reconstructedOutlineMesh = null;
        reconstructedOutlineMaterial = null;
        reconstructedOutlineVertices = null;
        reconstructedOutlineTriangles = null;
        reconstructedSilhouetteEdges = null;
        reconstructedTriangleFrontFacing = null;
        reconstructedSilhouetteLineIndices.Clear();
        nextReconstructedSilhouetteUpdateSeconds = -1.0f;
        if (meshDisplayButton != null) meshDisplayButton.gameObject.SetActive(false);
        if (meshOutlineMethodButton != null)
            meshOutlineMethodButton.gameObject.SetActive(false);
        if (alignmentRefineButton != null)
        {
            alignmentRefineButton.interactable = true;
            alignmentRefineButton.gameObject.SetActive(false);
            SetButtonLabel(alignmentRefineButton, "快速校准");
        }
        if (poseDiagnosticRecordButton != null)
        {
            poseDiagnosticRecordButton.interactable = true;
            poseDiagnosticRecordButton.gameObject.SetActive(false);
            SetButtonLabel(poseDiagnosticRecordButton, "录制位姿诊断");
        }
        if (meshDisplayModeText != null) meshDisplayModeText.text = "";
        if (meshOutlineMethodText != null) meshOutlineMethodText.text = "";
        RefreshMeshControlDockVisibility();
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
        // The target is retained only to report trajectory diagnostics.  It is
        // never used to accept or reject a frame in the server2 client.
        float distance = Mathf.Max(0.05f, assumedObjectDistanceMeters);
        poseDiversityTargetPosition =
            cameraPosition + cameraRotation * Vector3.forward * distance;
        poseDiversityTargetValid = true;
    }

    void MeasurePoseDiversityForAudit(
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

        if (acceptedPoseDiversityDirections.Count == 0)
        {
            minimumAngleDegrees = 180.0f;
            lastPoseDiversityMinimumAngle = -1.0f;
            return;
        }

        minimumAngleDegrees = 180.0f;
        foreach (Vector3 accepted in acceptedPoseDiversityDirections)
            minimumAngleDegrees = Mathf.Min(
                minimumAngleDegrees,
                Vector3.Angle(accepted, direction)
            );
        lastPoseDiversityMinimumAngle = minimumAngleDegrees;
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
        float poseDiversityMinimumAngle,
        string uploadEndpoint,
        string poseCoordinateFrame,
        string uploadSessionId,
        string refinementId,
        int uploadLifecycleGeneration
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
        form.AddField("pose_coordinate_frame", poseCoordinateFrame);
        form.AddField("session_id", uploadSessionId ?? "");
        form.AddField("refinement_id", refinementId ?? "");
        form.AddField(
            "lifecycle_generation",
            IntString(uploadLifecycleGeneration)
        );
        form.AddField(
            "capture_anchor_tracking_state",
            captureReferenceAnchor != null
                ? captureReferenceAnchor.trackingState.ToString()
                : "None"
        );
        form.AddField("screen_orientation", Screen.orientation.ToString());
        form.AddField("tracking_state", ARSession.state.ToString());
        form.AddField("display_matrix", MatrixString(displayMatrix));
        form.AddField("projection_matrix", MatrixString(projectionMatrix));
        form.AddField(
            "capture_view_policy",
            "fixed_interval_unfiltered_0p2s_v1"
        );
        form.AddField("capture_frame_filtering_applied", "false");
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
            "0"
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

        using (UnityWebRequest www = UnityWebRequest.Post(serverURL + uploadEndpoint, form))
        {
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success)
            {
                if (uploadEndpoint == "/alignment_refine/upload")
                    alignmentRefinementUploadedCount++;
                else
                {
                    acceptedPoseDiversityDirections.Add(poseDiversityDirection);
                    lastPoseDiversityMinimumAngle = poseDiversityMinimumAngle;
                }
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
